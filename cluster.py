"""
This module clusters the FFT frames produced by stft.py to
produce a code book and code words for quantization.
"""

import pickle as pkl
import time
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA

import conf
from util import ensure_dir


def cluster(data, num_clusters, cluster_type=conf.cluster_type):
    """
    Cluster the given data into the number of clusters specified by num_clusters.

    :param data: The data to cluster
    :param num_clusters: the number of clusters
    :param cluster_type: the clustering method to use. Can be "kmeans" for k means or 'mbk' for minibatch kmeans
    :returns:
    """
    if cluster_type is 'mbk':
        # mbk parameters:
        # see this stack overflow for reasoning on chosen reassignment ratio value
        # https://stackoverflow.com/questions/21447351/minibatchkmeans-parameters
        reassignment_ratio = 0
        batch_size = 10 * num_clusters
        init_size = 3 * num_clusters
        clust = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, init_size=init_size,
                                reassignment_ratio=reassignment_ratio)
        print('Running MiniBatch K-Means with K = {}...'.format(num_clusters))

    elif cluster_type is 'kmeans':
        clust = KMeans(n_clusters=num_clusters)
        print('Running K-Means with K = {}...'.format(num_clusters))

    else:
        raise IOError("Invalid cluster_type. Use cluster_type=\'kmeans\' for kmeans or"
                      "cluster_type=\'mbk\' for minibatch kmeans.")

    t0 = time.time()
    clust.fit(data)
    print('Fit in %s seconds' % (time.time() - t0))
    return clust.cluster_centers_, clust.labels_, clust.inertia_


# TODO: Quantize docstring
def quantize(data, code_book):
    """

    :param data:
    :param code_book:
    :return:
    """
    distances = pairwise_distances(X=data, Y=code_book)
    code = np.argmin(distances, axis=1)
    return pd.DataFrame(code)


# TODO: whiten docstring
def whiten(data, whiten_type='std'):
    if whiten_type is 'std':
        whitened = data.divide(data.std(axis=0, ddof=0), axis=1)
    if whiten_type is 'pca':
        pca = PCA(whiten=True)
        whitened = pd.DataFrame(pca.fit_transform(data))
    else:
        raise IOError("Invalid whiten_type. Use whiten_type=\'std\' for scaling by standard deviation or"
                      "whiten_type=\'pca\' for whitening with principle component analysis.")
    return whitened


def main(in_dir=conf.stft_path, out_dir=conf.cluster_path,
         k=conf.vocab_size, cluster_type=conf.cluster_type,
         return_inertia=False):

    ensure_dir(out_dir)

    # load spectrograms into dataframe
    num_files = 0
    data = pd.DataFrame()
    t_load = 0
    lengths = []
    names = []

    print("Reading from: %s" % out_dir)

    for filename in glob.glob(in_dir + '*.pkl'):
        # increment num_files and start timer
        num_files += 1
        t0 = time.time()

        print('Reading File #%d: %s' % (num_files, filename))

        # load pickle
        df = pkl.load(open(filename, "rb"))

        # add song name and frame num columns
        name = filename.split('/')[-1]
        name = name.split('.')[0]
        names.append(name)

        # append to data
        print('Appending %d data points...' % len(df))
        lengths.append(len(df))
        data = data.append(df, ignore_index=True)

        # end timer and add to total load time
        t1 = time.time()
        t_load += (t1 - t0)
        print('Done. (%f seconds)' % (t1 - t0))

    print('Finished loading all %d files.' % num_files)
    print('Total load time: %f seconds' % t_load)
    print('Total data points loaded: %d' % len(data))

    # whiten FFT frames before clustering
    if conf.whiten is not None:
        print('Whitening...')
        t0 = time.time()
        data = whiten(data, whiten_type=conf.whiten)
        print("Done. (%f seconds)" % (time.time() - t0))

    # ===>Clustering<===

    # cluster FFT frames
    codebook, labels, inertia = cluster(data, k, cluster_type)
    print("Distortion = %f" % inertia)

    codebook = pd.DataFrame(codebook)
    print('Pickling codebook...')
    pkl.dump(codebook, open(out_dir + "codebook.pkl", "wb"))
    print('Done.')

    # make dataframes for individual song labels
    lo = 0
    print('Pickling labels')
    for i in range(num_files):
        hi = lo + lengths[i]
        file_codes = pd.DataFrame(labels[lo:hi])
        lo = hi
        pkl.dump(file_codes, open(out_dir+"labels_{}.pkl".format(names[i]), "wb"))
    print('Done.')

    if return_inertia is True:
        return inertia


if __name__ == "__main__":
    main()
