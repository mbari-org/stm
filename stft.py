from scipy.io import wavfile
import math
import pandas as pd
import pickle as pkl
import glob
import librosa
from collections import OrderedDict
import time
import os
import numpy as np
from scipy.ndimage import gaussian_filter


def ensure_dir(fname):
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)


def read_wav(filepath, verbose=False):
    if verbose:
        print('Reading from %s' % filepath)
    fs, x = wavfile.read(filepath)
    if verbose:
        print('Done.')
    if verbose:
        print('Sample Rate: %d' % fs)
        print('Recording Length (s): %f' % (len(x) / fs))
        print('Number of Samples: %d' % len(x))
    return fs, x.astype(float)


def compute_stft(signal, window_size, overlap):
    stp = int(window_size * (1 - overlap))
    stft = np.divide(np.abs(librosa.stft(y=signal, n_fft=window_size, hop_length=stp)),
                     window_size)
    return stft


def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    normalized = np.divide(np.subtract(arr, mean), std)
    return normalized


def get_subset(stft, rng, num_freq_bins=16000):
    hz_per_bin = num_freq_bins / len(stft)
    lo_bin = math.floor(rng[0] / hz_per_bin)
    hi_bin = math.ceil(rng[1] / hz_per_bin)
    print('LO %d' % lo_bin)
    print('HI %d' % hi_bin)
    return stft[lo_bin:hi_bin + 1]


def run(in_dir, window_size, overlap, subset=None):

    # input and output directories
    out_dir = "./out/stft/win_{}/ovr_{}/sub_{}/".format(window_size, overlap, subset)
    ensure_dir(out_dir)

    # file number and time variables
    num_files = 0
    total_song_length = 0
    read_time = 0
    fft_time = 0
    df_conv_time = 0
    pkl_time = 0

    # for all songs in song_dir
    for filename in glob.glob(in_dir + '*.wav'):

        num_files += 1
        print('FILE NAME %s' % filename)
        print('FILE NUMBER %d' % num_files)

        # read wav and save song name
        print('Reading file...')
        t_0 = time.time()
        name = filename.split('/')[-1]
        name = name.split('.')[0]
        fs, x = read_wav(filename, verbose=False)
        total_song_length += len(x) / fs
        t_1 = time.time()
        read_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # compute spectrogram
        print('Computing spectrogram...')
        t_0 = time.time()
        stft = compute_stft(x, window_size, overlap)
        if type(subset) == tuple:
            stft = get_subset(stft, subset)
        t_1 = time.time()
        fft_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # TODO: add gaussian filter timer
        # gaussian filter
        print('Gaussian filtering...')
        stft = gaussian_filter(stft, sigma=2)
        print('Done.')

        # TODO: add normalization timer
        # normalization
        print('Normalizing...')
        stft = normalize(stft)
        print('Done.')

        # convert to dataframe
        print('Converting to dataframe...')
        t_0 = time.time()
        dic = OrderedDict()
        for i in range(len(stft)):
            dic['%d' % i] = stft[i]
        df = pd.DataFrame.from_dict(dic)
        t_1 = time.time()
        df_conv_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # dump dataframe as .pkl
        print('Pickling...')
        t_0 = time.time()
        pkl.dump(df, open(out_dir + name + '.pkl', "wb"))
        t_1 = time.time()
        pkl_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        if num_files % 25 == 0:
            print('DONE W/ %d FILES!' % num_files)
            print('--Read Time--')
            print('Total:%f seconds' % read_time)
            print('Avg: %f seconds' % (read_time / num_files))
            print('--Spectrogram Computing Time--')
            print('Total: %f seconds' % fft_time)
            print('Avg: %f seconds' % (fft_time / num_files))
            print('--Dataframe Conversion Time--')
            print('Total: %f seconds' % df_conv_time)
            print('Avg: %f seconds' % (df_conv_time / num_files))
            print('--Pickling Time--')
            print('Total: %f seconds' % pkl_time)
            print('Avg: %f seconds' % (pkl_time / num_files))

    print('FINISHED ALL %d FILES!' % num_files)
    print('--Read Time--')
    print('Total: %f seconds' % read_time)
    print('Avg: %f seconds' % (read_time / num_files))
    print('--Spectrogram Computing Time--')
    print('Total: %f seconds' % fft_time)
    print('Avg: %f seconds' % (fft_time / num_files))
    print('--Dataframe Conversion Time--')
    print('Total: %f seconds' % df_conv_time)
    print('Avg: %f seconds' % (df_conv_time / num_files))
    print('--Pickling Time--')
    print('Total: %f seconds' % pkl_time)
    print('Avg: %f seconds' % (pkl_time / num_files))
    print('Total Runtime: %f seconds' % (read_time + fft_time + df_conv_time + pkl_time))

    return out_dir

