"""
This module clusters the FFT frames produced by stft.py to
produce a code book and code words for quantization.
"""

import pickle as pkl
import time
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA

import conf


def cluster(data, num_clusters, cluster_type=conf.cluster_type):
    """
    Cluster the given data into the number of clusters specified by num_clusters.

    :param data: The data to cluster
    :param num_clusters: the number of clusters
    :param cluster_type: the clustering method to use. Can be "kmeans" for k means or 'mbk' for minibatch kmeans
    :returns:
    """
    if cluster_type == 'mbk':
        # mbk parameters:
        # see this stack overflow for reasoning on chosen reassignment ratio value
        # https://stackoverflow.com/questions/21447351/minibatchkmeans-parameters
        reassignment_ratio = 0
        batch_size = 10 * num_clusters
        init_size = 3 * num_clusters
        clust = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, init_size=init_size,
                                reassignment_ratio=reassignment_ratio)
        print(f'Running MiniBatch K-Means with K = {num_clusters}...')
    elif cluster_type == 'kmeans':
        clust = KMeans(n_clusters=num_clusters)
        print(f'Running K-Means with K = {num_clusters}...')
    else:
        raise IOError("Invalid cluster_type. Use cluster_type='kmeans' for kmeans or"
                      "cluster_type='mbk' for minibatch kmeans.")

    t0 = time.time()
    clust.fit(data)
    print(f'Fit in {time.time() - t0} seconds')
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
    if whiten_type == 'std':
        whitened = data.divide(data.std(axis=0, ddof=0), axis=1)
    elif whiten_type == 'pca':
        pca = PCA(whiten=True)
        whitened = pd.DataFrame(pca.fit_transform(data))
    else:
        raise IOError("Invalid whiten_type. Use whiten_type='std' for scaling by standard deviation or"
                      "whiten_type='pca' for whitening with principle component analysis.")
    return whitened


def main(in_dir=conf.stft_path, out_dir=conf.cluster_path,
         k=conf.vocab_size, cluster_type=conf.cluster_type,
         return_inertia=False):

    out_dir.mkdir(parents=True, exist_ok=True)

    # load spectrograms into dataframe
    num_files = 0
    data_list = []
    t_load = 0
    lengths = []
    names = []

    print(f"Reading from: {out_dir}")

    for filename in in_dir.rglob('*.pkl'):
        # increment num_files and start timer
        num_files += 1
        t0 = time.time()

        print(f'Reading File #{num_files}: {filename}')

        df = pkl.load(open(filename, "rb"))

        # add song name and frame num columns
        name = filename.stem
        names.append(name)

        print(f'Appending {len(df)} data points...')
        lengths.append(len(df))
        data_list.append(df)

        # end timer and add to total load time
        t1 = time.time()
        t_load += (t1 - t0)
        print(f'Done. ({t1 - t0:.6f} seconds)')

    data = pd.concat(data_list, ignore_index=True)

    print(f'Finished loading all {num_files} files.')
    print(f'Total load time: {t_load:.6f} seconds')
    print(f'Total data points loaded: {len(data)}')

    # whiten FFT frames before clustering
    if conf.whiten is not None:
        print('Whitening...')
        t0 = time.time()
        data = whiten(data, whiten_type=conf.whiten)
        print(f"Done. ({time.time() - t0:.6f} seconds)")

    # ===>Clustering<===

    # cluster FFT frames
    codebook, labels, inertia = cluster(data, k, cluster_type)
    print(f"Distortion = {inertia:.6f}")

    codebook = pd.DataFrame(codebook)
    print('Pickling codebook...')
    with open(out_dir / 'codebook.pkl', 'wb') as f:
        pkl.dump(codebook, f)
    print('Done.')

    # make dataframes for individual song labels
    lo = 0
    print('Pickling labels')
    for i in range(num_files):
        hi = lo + lengths[i]
        file_codes = pd.DataFrame(labels[lo:hi])
        lo = hi
        with open(out_dir / f'labels_{names[i]}.pkl', 'wb') as f:
            pkl.dump(file_codes, f)
    print('Done.')

    if return_inertia is True:
        return inertia


if __name__ == "__main__":
    main()