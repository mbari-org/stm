from datetime import datetime
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import math
import pandas as pd
import pickle as pkl

import conf


def spectrogram(stft, frequencies, window_size, overlap, fs,
                y='linear', start_time=None, c_bar=None):

    hop_len = window_size * (1 - overlap)
    display.specshow(stft, y_axis=y, x_axis='time',
                     sr=fs, hop_length=hop_len,
                     y_coords=frequencies,
                     cmap="Blues")
    ax = plt.gca()
    num_frames = stft.shape[1]

    def time_formatter(x, pos):
        """Convert seconds to MM:SS.ms format"""
        if start_time is not None:
            time_in_ms = int(x * 1000)
            time = start_time + pd.Timedelta(milliseconds=time_in_ms)
            return time.strftime('%M:%S.%f')[:-3]
        else:
            return f"{int(x // 60):02}:{int(x % 60):02}.{int((x % 1) * 1000):03}"

    minor_tick_interval = num_frames // 10 if num_frames >= 10 else 1
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
    ax.xaxis.set_minor_formatter(FuncFormatter(time_formatter))
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    if isinstance(c_bar, str):
        plt.colorbar(format=f"%.2f {c_bar}")
    return plt.gca()

def stacked_bar(data, legend: str = None):
    data_length = len(data)
    data_height = len(data[0])
    P = []
    ind = np.arange(data_length)
    bottom = np.zeros(len(data))
    width = 1

    for z in range(data_height):
        col = [row[z] for row in data]
        P.append(plt.bar(x=ind, height=col, width=width, bottom=bottom, align='edge'))
        bottom = np.add(col, bottom)
    plt.xlim(xmin=0, xmax=len(data))
    plt.legend(
            tuple(_[0] for _ in P),
            [f"{legend}{i}" for i in range(data_height)],
            loc="lower center",
            ncol=10
        )

# TODO: Add input checking and option to plot the whole spectrogram
def main(times=conf.times, model_path=conf.model_path, stft_path=conf.stft_path, doc_path=conf.doc_path,
         target_file=conf.target_file, window_size=conf.window_size, overlap=conf.overlap, gamma=conf.gamma,
         fs=conf.sample_rate, subset=conf.subset, words_per_doc=conf.words_per_doc, use_pcen=conf.use_pcen):

    # Get the lookup file to find the target file within the documents
    lookup_file = doc_path / "lookup.csv"
    if not lookup_file.exists():
        print(f"Lookup file {lookup_file} does not exist.")
        return

    # Find the row with the target_file in the lookup file
    target_file_path = doc_path /  f"{target_file}.csv"
    lookup_df = pd.read_csv(lookup_file)
    target_row = lookup_df[lookup_df['filename'] == target_file_path.name]
    if target_row.empty:
        print(f"Target file {target_file_path} not found in lookup file.")
        return
    start_ts = target_row.ms_start.values[0]
    end_ts = target_row.ms_end.values[0]

    theta_model = pd.read_csv(model_path / "theta.csv", header=None).values
    ms_per_doc = int(((window_size / fs) * (1 - overlap) * 1000)) * words_per_doc
    theta = theta_model[start_ts//ms_per_doc:end_ts - ms_per_doc//ms_per_doc]
    stft = pkl.load(open(stft_path / f'{target_file}.pkl', "rb"))

    secs_per_frame = window_size * (1 - overlap) / fs
    secs_per_doc = secs_per_frame * words_per_doc

    # Write out a TSV with the Raven compatible format
    # Selection View Channel Begin Time (s) End Time (s) Topic Probability
    # Where View is 1, Selection indexes from 0 and Topic indexes from 0
    with open(model_path / f"raven_topics_top1_{target_file_path.stem}.txt", "w") as f:
        f.write("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tTopic\tTopic Probability\n")
        for i, d in enumerate(theta):
            start_t = i * secs_per_doc
            end_t = start_t + secs_per_doc
            # Get the max topic for this document and write it out
            top_1_topic = np.argmax(d)
            top_1_prob = d[top_1_topic]
            f.write(f"{i+1}\t1\t1\t{start_t:.5f}\t{end_t:.5f}\t{int(top_1_topic)}\t{top_1_prob:.5f}\n")

    with open(model_path / f"raven_topics_top2_{target_file_path.stem}.txt", "w") as f:
        f.write("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tTopic 1\tTopic 1 Probability\tTopic 2\tTopic 2 Probability\n")
        for d in theta:
            start_t = i * secs_per_doc
            end_t = start_t + secs_per_doc
            # Get the top 2 topics for this document and write them out
            top_2_topic = np.argsort(d)[-2:][::-1]  # Get the indices of the top 2 topics
            top_2_prob = d[top_2_topic]
            if len(top_2_topic) < 3:
                f.write(f"{i+1}\t1\t1\t{start_t:.5f}\t{end_t:.5f}\t{int(top_2_topic[0])}\t{top_2_prob[0]:.5f}\t{int(top_2_topic[1])}\t{top_2_prob[1]:.5f}\n")

    frequencies = stft.index.values
    stft = stft.values

    if times:
        start_time = pd.to_datetime(times[0], unit='s')
        end_time = pd.to_datetime(times[1], unit='s')
        prefix = f"{target_file_path.stem}_topics_{start_time.strftime('%M%S')}_{end_time.strftime('%M%S')}_"
    else:
        start_time = pd.to_datetime(0, unit='s')
        prefix = f"{target_file_path.stem}_topics_"

    if times is not None:
        start_doc = math.floor(times[0] / secs_per_doc)
        end_doc = math.ceil(times[1] / secs_per_doc)
        real_start_t = secs_per_doc * start_doc
        real_end_t = secs_per_doc * end_doc
        theta = theta[start_doc: end_doc]

        start_frame = int(real_start_t * (1 / secs_per_frame))
        end_frame = int(real_end_t * (1 / secs_per_frame))
        stft = stft[:,start_frame: end_frame]

    fig = plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    spectrogram(stft, frequencies, window_size, overlap, fs, y='linear', start_time=start_time)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"{prefix}{timestamp}.png"
    plt.title(final_file)
    plt.suptitle(f"alpha={conf.alpha}, beta={conf.beta}, gamma={gamma}, g_time={conf.g_time}, "
                 f"vocab_size={conf.vocab_size}, words_per_doc={words_per_doc}, "
                 f"window_size={window_size}, overlap={overlap},  "
                 f"fs={fs}, subset={subset} use_pcen={use_pcen}" )
    plt.subplot(2, 1, 2)
    stacked_bar(theta, legend="T")
    plt.xlabel("Documents")
    plt.ylabel("Topic Probability")

    plt.tight_layout()
    fig.savefig(model_path / final_file, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()