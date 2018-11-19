from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle as pkl

import conf
from util import map_range
from stft import read_wav, compute_stft, get_subset, normalize_stft


def spectrogram(stft, window_size, overlap, fs,
                y='linear', freq_subset: tuple = None, c_bar=None):

    hop_len = window_size * (1 - overlap)
    display.specshow(stft, y_axis=y, x_axis='time',
                     sr=fs, hop_length=hop_len)

    if c_bar is str:
        plt.colorbar(format="%.2f "+"{}".format(c_bar))

    if freq_subset:
        hz_per_bin = (fs / 2) / (1 + window_size / 2)
        locs, labels = plt.yticks()
        c = hz_per_bin*math.floor(freq_subset[0]/hz_per_bin)
        d = hz_per_bin*math.ceil(freq_subset[1]/hz_per_bin)
        new_labels = ["%.2f" % map_range(locs[i], locs[0], locs[-1], c, d) for i in range(len(locs))]
        plt.yticks(locs, new_labels)

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

    if legend:
        plt.legend(tuple(_[0] for _ in P),
                   ["{}{}".format(legend, i) for i in range(data_height)],
                   loc='lower right')

    return plt.gca()

def visualize_preproc():
    wav_file = '/Users/thomasbergamaschi/Code/hb-song-analysis/dataset/HBSe_20151207T070326.wav'

    fs, x = read_wav(wav_file, verbose=True)

    fs = 32000
    window_size = 4096
    ovr = 0.9
    hop_len = window_size*(1-ovr)
    subset = (50, 2000)
    hz_per_bin = (fs/2) / (1+window_size/2)


    # Waveform
    fig = plt.figure(figsize=(10, 3))
    display.waveplot(x, sr=fs)
    plt.title('Waveform (30 kHz Sample Rate)')
    plt.tight_layout()
    plt.show()


    # Spectrogram
    fig = plt.figure(figsize=(12, 3))
    stft = compute_stft(x, window_size, ovr)
    stft = librosa.amplitude_to_db(stft, ref=np.max)
    display.specshow(stft, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram ({} Window, {} Overlap)'.format(window_size, ovr))
    plt.tight_layout()
    plt.show()


    # Subsetted Spectrogram
    fig = plt.figure(figsize=(12, 3))
    stft = get_subset(stft, rng=(50, 2000))
    display.specshow(stft, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Spectrogram (50Hz to 2kHz Subset)')
    plt.tight_layout()
    plt.show()


    # Gaussian Filtering
    fig = plt.figure(figsize=(12, 3))
    blurred = gaussian_filter(stft, sigma=2)
    display.specshow(blurred, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Gaussian Filter (sigma=2)')
    plt.tight_layout()
    plt.show()


    # Normalization
    fig = plt.figure(figsize=(12, 3))
    normal = normalize_stft(blurred)
    display.specshow(normal, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f std')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Normalized')
    plt.tight_layout()
    plt.show()


# TODO: Add input checking and option to plot the whole spectrogram
def main(times=conf.times, model_path=conf.model_path, stft_path=conf.stft_path,
         target_file=conf.target_file, window_size=conf.window_size, overlap=conf.overlap,
         fs=conf.sample_rate, subset=conf.subset, words_per_doc=conf.words_per_doc):

    theta = pd.read_csv(model_path + "theta.csv", header=None).values
    stft = pkl.load(open(stft_path+target_file+'.pkl', "rb"))

    secs_per_frame = window_size * (1 - overlap) / fs
    secs_per_doc = secs_per_frame * words_per_doc

    if times is not None:
        start_doc = math.floor(times[0] / secs_per_doc)
        end_doc = math.ceil(times[1] / secs_per_doc)
        real_start_t = secs_per_doc * start_doc
        real_end_t = secs_per_doc * end_doc
        theta = theta[start_doc: end_doc]

        start_frame = int(real_start_t * (1 / secs_per_frame))
        end_frame = int(real_end_t * (1 / secs_per_frame))
        stft = stft[start_frame:end_frame]

    stft = np.matrix.transpose(stft.values)

    fig = plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    spectrogram(stft, window_size, overlap, fs,
                freq_subset=subset)

    plt.subplot(2, 1, 2)
    stacked_bar(theta, legend="T")
    plt.xlabel("Documents")
    plt.ylabel("Topic Probability")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
