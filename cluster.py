import pickle as pkl
import time
import pandas as pd
import glob
import os
from sklearn.cluster import MiniBatchKMeans


def ensure_dir(fname):
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)


def main():
    # ===>Load data<====
    in_dir = './out/stft_win_1024_ovr_50_50hz_5000hz/'
    out_dir = './out/clustering/'

    num_files = 0
    data = pd.DataFrame()
    t_load = 0
    lengths = []
    names = []

    for pkl_file in glob.glob(in_dir + '*.pkl'):
        # increment num_files and start timer
        num_files += 1
        t0 = time.time()

        print('Reading File #%d: %s' % (num_files, pkl_file))

        # load pickle
        df = pkl.load(open(pkl_file, "rb"))

        # add song name and frame num columns
        name = pkl_file.split('/')[-1]
        name = name.split('.')[0]
        name = name.split('_')[0] + '_' + name.split('_')[1]
        df['song_name'] = [name] * len(df)
        df['frame_num'] = list(range(len(df)))
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

    # ===>Minibatch K-Means<===

    # k values and parameters
    k = 100  # [1000, 10000, 20000, 30000, 40000, 50000]
    batch_size = 100

    # run minibatch k-means
    init_size = 3 * k
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, init_size=init_size)
    print('Running MiniBatch K-Means with K = {}...'.format(k))
    t0 = time.time()
    mbk.fit(data.drop(['song_name', 'frame_num'], axis=1))
    print('Fit in %s seconds' % (time.time() - t0))

    # make directories
    labels_dir = out_dir+'/labels/'
    ensure_dir(out_dir)
    ensure_dir(labels_dir)

    # centroids dataframe
    codebook = pd.DataFrame(mbk.cluster_centers_)
    print('Pickling codebook...')
    pkl.dump(codebook, open(out_dir + "K_{}_codebook.pkl".format(k), "wb"))
    print('Done.')

    # labels dataframes for each song
    lo = 0
    print('Pickling labels')
    for i in range(num_files):
        hi = lo + lengths[i]
        labels = pd.DataFrame({'labels': mbk.labels_[lo:hi]})
        lo = hi
        pkl.dump(labels, open(labels_dir + "labels_{}.pkl".format(names[i]), "wb"))
    print('Done.')


if __name__ == '__main__':
    main()
