import pickle as pkl
import time
import pandas as pd
import glob
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


def ensure_dir(fname):
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)


def run(in_dir, k, minibatch=False):
    # ===>Load data<====
    out_dir = in_dir+"k_{}/".format(k)

    num_files = 0
    data = pd.DataFrame()
    t_load = 0
    lengths = []
    names = []

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

    # ===>Normalization<===
    print('Normalizing...')

    t0 = time.time()
    # compute mean and std
    mean = data.mean()
    std = data.std()

    # subtract mean and divide by std
    normed_data = data.subtract(mean)
    normed_data = normed_data.divide(std)

    print("Done. (%f seconds)"%(time.time() - t0))

    # ===>Minibatch K-Means<===


    if minibatch:
        # mbk parameters:
        # see this stack overflow for reasoning on chosen reassignment ratio value
        # https://stackoverflow.com/questions/21447351/minibatchkmeans-parameters
        reassignment_ratio = 0
        batch_size = 10 * k
        init_size = 3 * k
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, init_size=init_size,
                                 reassignment_ratio=reassignment_ratio)
        print('Running MiniBatch K-Means with K = {}...'.format(k))

    else:
        kmeans = KMeans(n_clusters=k)
        print('Running K-Means with K = {}...'.format(k))


    t0 = time.time()
    kmeans.fit(normed_data)
    print('Fit in %s seconds' % (time.time() - t0))

    # make directories
    labels_dir = out_dir+'/labels/'
    ensure_dir(out_dir)
    ensure_dir(labels_dir)

    # centroids dataframe
    codebook = pd.DataFrame(kmeans.cluster_centers_)
    print('Pickling codebook...')
    pkl.dump(codebook, open(out_dir + "codebook.pkl", "wb"))
    print('Done.')

    # labels dataframes for each song
    lo = 0
    print('Pickling labels')
    for i in range(num_files):
        hi = lo + lengths[i]
        labels = pd.DataFrame({'labels': kmeans.labels_[lo:hi]})
        lo = hi
        pkl.dump(labels, open(labels_dir + "labels_{}.pkl".format(names[i]), "wb"))
    print('Done.')

    return out_dir


