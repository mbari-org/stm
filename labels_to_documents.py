import pickle as pkl
import glob
import numpy as np

import conf
from util import write_csv
from util import ensure_dir


def group_docs(words, words_per_doc, window_size, overlap, start_t=0, fs=32000):
    ms_per_word = int(((window_size / fs) * (1-overlap) * 1000))
    docs = []

    i = 0
    j = words_per_doc
    while i < len(words):
        timestamp = (j * ms_per_word) + start_t
        d = [timestamp]
        d.extend(list(words[0][i:j]))
        docs.append(d)
        i += words_per_doc
        if (j + words_per_doc) <= len(words):
            j += words_per_doc
        else:
            j = len(words)
    return docs


def label_docs(doc_times, event_times, event_labels):
    j = 0  # docs index
    doc_labels = np.zeros(len(doc_times))

    for i in range(len(event_times)):
        # start and end of event
        t0 = event_times.loc[i][0]
        t1 = event_times.loc[i][1]

        while doc_times[j] <= t0:
            j += 1  # increment doc

        while doc_times[j] <= t1:
            doc_labels[j] = event_labels.loc[i]
            j += 1  # increment doc

        doc_labels[j] = event_labels.loc[i]
        j += 1

    return doc_labels


def main(in_dir=conf.cluster_path, out_dir=conf.doc_path,
         words_per_doc=conf.words_per_doc, window_size=conf.window_size, overlap=conf.overlap, fs=32000):

    ensure_dir(out_dir)

    for filename in glob.glob(in_dir + 'labels_*.pkl'):

        name = filename.split('/')[-1]
        name = name.split('.')[0]
        name = name.split('_')[-2]+'_'+name.split('_')[-1]

        print("Making documents with {} words for {}...".format(words_per_doc, name))

        labels = pkl.load(open(filename, "rb"))

        docs = group_docs(words=labels, words_per_doc=words_per_doc, window_size=window_size, overlap=overlap)
        """
        ms_per_word = int(((window_size / fs) * overlap) * 1000)
        docs = []

        i = 0
        j = words_per_doc
        while i < len(labels):
            timestamp = j*ms_per_word
            d = [timestamp]
            d.extend(labels[0].values[i:j])
            docs.append(d)
            i += words_per_doc
            if (j+words_per_doc) <= len(labels):
                j += words_per_doc
            else:
                j = len(labels)
        """

        write_csv(docs, out_dir + name + ".csv")

        print('Done.')


if __name__ == "__main__":
    main()
