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


def main():

    # parameters
    fs = 32000
    frame_len = 1024
    overlap = 0.5
    doc_len = 32

    in_dir = './out/clustering/labels/'
    out_dir = './out/documents/'

    for pkl_file in glob.glob(in_dir + '*.pkl'):

        filename = pkl_file.split('/')[-1]
        filename = filename.split('.')[0]
        filename = filename.split('_')[-2]+'_'+filename.split('_')[-1]

        print("Making documents for {}...".format(filename))

        labels = pkl.load(open(pkl_file, "rb"))

        ms_per_word = int(((frame_len/fs)*overlap)*1000)
        docs = []

        i = 0
        j = doc_len
        while i < len(labels):
            timestamp = j*ms_per_word
            d = [timestamp]
            d.extend(list(labels['labels'][i:j]))
            docs.append(d)
            i += doc_len
            if (j+doc_len) <= len(labels):
                j += doc_len
            else:
                j = len(labels)

        write_csv(docs, out_dir+"doc_"+str(doc_len)+"_"+filename+".csv")

        print('Done.')


if __name__ == "__main__":
    main()

