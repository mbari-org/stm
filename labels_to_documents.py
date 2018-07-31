import pickle as pkl
import csv
import os
import glob


def ensure_dir(fname):
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)


def write_csv(docs, fname):
    ensure_dir(fname)
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(docs)


def run(in_dir, words_per_doc, window_size, overlap, fs=32000):

    out_dir = './out/model/{}_word_docs/'.format(words_per_doc)

    for filename in glob.glob(in_dir + '*.pkl'):

        name = filename.split('/')[-1]
        name = name.split('.')[0]
        name = name.split('_')[-2]+'_'+name.split('_')[-1]

        print("Making documents with {} words for {}...".format(words_per_doc, name))

        labels = pkl.load(open(filename, "rb"))

        ms_per_word = int(((window_size / fs) * overlap) * 1000)
        docs = []

        i = 0
        j = words_per_doc
        while i < len(labels):
            timestamp = j*ms_per_word
            d = [timestamp]
            d.extend(list(labels['labels'][i:j]))
            docs.append(d)
            i += words_per_doc
            if (j+words_per_doc) <= len(labels):
                j += words_per_doc
            else:
                j = len(labels)

        write_csv(docs, out_dir+name+".csv")

        print('Done.')

    return out_dir

