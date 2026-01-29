import csv
from pathlib import Path

import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phraser
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

import conf
from gensim.models import Phrases

from visualize import spectrogram, stacked_bar, phrase_plot

topic_word = ["click", "whistle", "chirp", "squawk", "song", "call", "noise", "sound", "vocalization",
              "birdsong", "tweet", "callnote", "alarm call", "song phrase", "song type", "song component",
              "song unit", "song element", "song note", "song syllable", "song motif", "song pattern",
              "song sequence", "song structure", "song form", "song style", "song dialect", "song variation",
              "song repertoire", "song learning", "song imitation", "song development", "song evolution",
              "song communication", "song function", "song meaning", "song context", "song behavior",]

def grouped_ranges(df):
    """
    Group the DataFrame by contiguous segments of the same best topic.
    Creates two new columns called "range" and "topic_word" in the data frame
    """
    df["best_topic"] = df.apply(lambda x: np.argmax(x.values), axis=1)
    group_change = (df['best_topic'] != df['best_topic'].shift()).cumsum()
    grouped = df.groupby(group_change)

    ranges = {}
    for _, group in grouped:
        start = group.index[0]
        end = group.index[-1]
        for idx in group.index:
            if end - start < 2:  # If the segment is too short, mark it as (-1, -1)
                ranges[idx] = (-1, -1)
            else:
                ranges[idx] = (start, end)

    # Add the range column
    df['range'] = df.index.map(ranges)

    good_word = (df['range'] != (-1, -1))
    # Apply topic_word to the best topic only if the range is valid (not (-1, -1)) otherwise use "and
    df["topic_word"] = np.where(good_word, df["best_topic"].apply(lambda idx: topic_word[int(idx)]), "and")

    # Group by the range and take the first topic_word in each range with the number of occurrences, e.g. 3 clicks
    df = df.groupby('range').agg({
        'topic_word': 'first',
    }).reset_index()
    # df['range'] = df['range'].apply(lambda x: f"{x[0]}-{x[1]}")
    def best_topic_decor(row):
        return row['topic_word']
        # if row['topic_word'] == "and":
        #     return "and"
        # r = row["range"].split('-')
        # num_units = (int(r[1]) - int(r[0]) + 1)
        # # Convert the number of units to a string, e.g. "3 clicks" to "three clicks"
        # if num_units == 1:
        #     return f"{row['topic_word']}"
        # elif num_units == 2:
        #     return f"few {row['topic_word']}"
        # elif 2 < num_units < 5:
        #     number = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        #               "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        #               "eighteen", "nineteen", "twenty"]
        #     return f"some {row['topic_word']}"
        # else:
        #     return f"many {row['topic_word']}"

    df['best_topic'] = df.apply(best_topic_decor, axis=1)
    return df

def main(out_dir=conf.model_path, target_file=conf.target_file_single, doc_path=conf.doc_path,
         stft_path=conf.stft_path,
         window_size=conf.window_size, overlap=conf.overlap,
         fs=conf.sample_rate):

    model_path = Path(out_dir)
    doc_path = Path(doc_path)

    # Get the topic model
    topic_model = model_path / "topics.csv"
    if not topic_model.exists():
        print(f"Topic model file {topic_model} does not exist.")
        return

    # Get the lookup file
    lookup_file = doc_path / "lookup.csv"
    if not lookup_file.exists():
        print(f"Lookup file {lookup_file} does not exist.")
        return

    theta_df = pd.read_csv(model_path / "theta.csv")

    # Add column headers called T0, T1, etc. for each column
    theta_df.columns = [f"T{i}" for i in np.arange(theta_df.shape[1])]

    theta_df = grouped_ranges(theta_df)

    from collections import Counter

    def find_repeated_sequences(data, n=3):
        phrases = [tuple(data[i:i + n]) for i in range(len(data) - n + 1)]
        return Counter(phrases).most_common()

    sequences = find_repeated_sequences(theta_df["best_topic"].values, n=1)
    print(f"Most common {len(sequences)} sequences:")
    for seq, count in sequences:
        print(f"Sequence: {seq}, Count: {count} ")

    # Drop any sequences that have a count of 1
    sequences = [seq[0] for seq, count in sequences if count > 1]
    print(f"Filtered sequences: {len(sequences)} sequences with count > 1")

    # Create a document corpus from the topic phrases that have words in the common sequences
    sound_units = theta_df["topic_word"].values
    sound_units = [unit for unit in sound_units if unit in sequences]
    sound_units = [unit for unit in sound_units if unit not in "and"]
    print(f"Filtered sound units: {len(sound_units)} sound units")

    # Run the phrase model
    phrase_model = Phrases([sound_units], min_count=2, threshold=0.0005, connector_words=ENGLISH_CONNECTOR_WORDS)
    for sent in phrase_model[sound_units]:
        pass

    frozen_phrases = phrase_model.freeze()
    print(f"Frozen phrases: {frozen_phrases.phrasegrams}")
    phraser = Phraser(phrase_model)

    doc = sound_units
    phrased_doc = phraser[doc]

    spans = []
    i = 0
    j = 0

    while i < len(doc):
        token = phrased_doc[j]
        if "_" in token:
            parts = token.split("_")
            span_length = len(parts)
            spans.append((i, i + span_length - 1, token))
            i += span_length
        else:
            i += 1
        j += 1

    with open(model_path / "phrases_with_positions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "phrase"])

        for start, end, phrase in spans:
            writer.writerow([start, end, phrase])

    phrases = pd.read_csv(model_path / "phrases_with_positions.csv")

    # Get the unique phrases in the column "phrase"
    unique_phrases = phrases["phrase"].unique()
    num_rows = len(theta_df)
    phrase_df = pd.DataFrame(0, index=np.arange(num_rows), columns=np.arange(len(unique_phrases) + 1))
    unique_phrases_list = unique_phrases.tolist()
    for i, row in phrases.iterrows():
        start_index = row["start"]
        end_index = row["end"]
        # Get the index of the unique_phrases_list that matches the row["phrase"]
        phrase_index = unique_phrases_list.index(row["phrase"])
        phrase_df.loc[start_index:end_index, phrase_index + 1] = 1 # column 0 is reserved for no phrase

    stft = pkl.load(open(stft_path / f'{target_file}.pkl', "rb"))
    stft = np.matrix.transpose(stft.values)

    fig = plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    spectrogram(stft, window_size, overlap, fs)

    plt.subplot(2, 1, 2)
    stacked_bar(phrase_df.values, legend="T")
    plt.xlabel("Documents")
    plt.ylabel("Topic Probability")

    plt.tight_layout()
    plt.show()














if __name__ == "__main__":
    main()