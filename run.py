import glob
import pandas as pd
import numpy as np

import stft
import cluster
import labels_to_documents
import topic_model
import conf as cnf

# TODO: change file types from pickle to JSON


def preprocessing():

    in_dir = cnf.WAVPATH

    # stft parameters to iterate over
    window_size = [1024]
    overlap = [0.5]
    subset = (50, 2000)

    # kmeans parameters
    k_vals = [1000]

    for win in window_size:
        for ovr in overlap:
            stft_dir = stft.run(in_dir, window_size=win, overlap=ovr, subset=subset)
            for k in k_vals:
                cluster.run(stft_dir, k, minibatch=True)


def doc_building():

    # ===>Build Documents<===

    # document parameters
    words_per_doc = 32
    window_size = 1024
    overlap = 0.5
    subset = (50, 2000)
    k_val = 1000
    labels_dir = "./out/stft/win_{}/ovr_{}/sub_{}/k_{}/labels/".format(window_size, overlap, subset, k_val)

    docs_dir = labels_to_documents.run(in_dir=labels_dir, words_per_doc=words_per_doc,
                                       window_size=window_size, overlap=overlap)

    print(docs_dir)


def modeling():

    # ===>Run ROST<===

    # model parameters
    filename = 'HBSe_20151207T070326'
    words_per_doc = 32
    in_dir = './out/model/{}_word_docs/'.format(words_per_doc)

    vocab_size = 1000
    num_topics = 5
    alpha = 0.1
    beta = 0.001
    g_time = 2
    cell_space = 0


    # online parameters
    online = False
    online_mint = 5

    topics_dir = topic_model.run(in_dir=in_dir, target_file=filename, W=vocab_size, T=num_topics,
                                 alpha=alpha, beta=beta,
                                 g_time=g_time, cell_space=cell_space,
                                 online=online, online_mint=online_mint)


def classification():

    train_dir = '/Users/bergamaschi/Documents/HumpbackSong/labeled_by_miriam/train'
    test_dir = '/Users/bergamaschi/Documents/HumpbackSong/labeled_by_miriam/test/'

    window_size = 1024
    overlap = 0.5
    subset = (50, 2000)
    k = 1000
    words_per_doc = 32

    train_data = pd.DataFrame()
    train_names = pd.DataFrame()
    test_data = pd.DataFrame()
    test_names = pd.DataFrame()

    # ===>Pre-processing<===

    # For training set:
    for fname in glob.glob(train_dir+"*.wav"):
        # Read wav
        fs, signal = stft.read_wav(fname, verbose=True)

        # Compute stft and subset
        Sxx = stft.compute_stft(signal, window_size, overlap, fs, subset)

        # Gaussian filter
        filtered = stft.gaussian_filter(Sxx, sigma=2)

        # convert to DataFrame
        data_df = stft.stft_to_dataframe(filtered)

        # create dataframe for name and frame number for later use
        name = fname.split('.')[0]
        name = name.split('/')[-1]
        name_df = pd.DataFrame({'name': [name]*len(data_df),
                                'frame_num': list(range(len(data_df)))})

        # append to train data
        train_data = train_data.append(data_df, ignore_index=True)
        train_names = train_names.append(name_df, ignore_index=True)

    # Normalize
    train_normalized, train_mean, train_std = stft.normalize(train_data)
    #   Cluster and save code book/labels
    code_book, train_words = cluster.cluster(train_normalized, k=k, minibatch=True)

    #   Group words into documents
    train_docs = {}
    train_labels = {}
    names = list(set(train_names['name']))
    for n in names:
        words = train_words.loc[train_names['name'] == n]
        docs = labels_to_documents.group_docs(words,
                                              words_per_doc=words_per_doc, window_size=window_size,
                                              overlap=overlap, fs=fs)

    #   Label documents



    # For testing set:
    for fname in glob.glob(test_dir+"*.wav"):
        # Read wav
        fs, signal = stft.read_wav(fname, verbose=True)

        # Compute stft and subset
        Sxx = stft.compute_stft(signal, window_size, overlap, fs, subset)

        # Gaussian filter
        filtered = stft.gaussian_filter(Sxx, sigma=2)

        # convert to DataFrame
        data_df = stft.stft_to_dataframe(filtered)

        # create dataframe for name and frame number for later use
        name = fname.split('.')[0]
        name = name.split('/')[-1]
        name_df = pd.DataFrame({'name': [name]*len(data_df),
                                'frame_num': list(range(len(data_df)))})

        # append to test data and names
        test_data = test_data.append(data_df, ignore_index=True)
        test_names = test_names.append(name_df, ignore_index=True)


    # Normalize using training avg/std
    test_normalized, mean, std = stft.normalize(test_data, mean=train_mean, std=train_std)
    #   Quantize using training code book
    test_words = cluster.quantize(test_normalized, code_book=code_book)
    #   Group labels into documents
    #   Label documents

    # ===>Training<===

    # Run ROST on training documents
    # Compute marginal distribution: P(class | k = z)
    # Plot marginal distribution
    # Compute conditional distribution: P(k = z | class)
    # Assign topics their highest probability label

    # ===>Testing<===

    # Run ROST on testing docs using:
    #   --in.topicmodel <in.topicmodel.csv>
    #   --topicmodel.update=0

    # Compute P(z=k | d)

if __name__ == '__main__':
    # preprocessing()
    # doc_building()
    # modeling()
    #classification()




