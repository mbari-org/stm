"""
This module computes spectrograms from given wav files that
are preprocessed in the manner specified by conf.py
"""

from scipy.io import wavfile
import math
import pandas as pd
import pickle as pkl
import glob
import librosa
from collections import OrderedDict
import time
import numpy as np
from scipy.ndimage import gaussian_filter

import conf
from util import ensure_dir


def read_wav(fpath, verbose=False):
    """
    Reads a .wav file and return the signal and sample rate.

    :param fpath: file path of .wav file
    :param verbose: if true, read_wav prints information about the .wav
    :returns: the tuple (fs, x) where fs is the sample rate and x is the data read
    """
    if verbose:
        print('Reading from %s' % fpath)
    fs, signal = wavfile.read(fpath)
    if verbose:
        print('Done.')
        print('Sample Rate: %d' % fs)
        print('Recording Length (s): %f' % (len(signal) / fs))
        print('Number of Samples: %d' % len(signal))
    return fs, signal.astype(float)


def compute_stft(signal, window_size, overlap, power=False):
    """
    Compute the spectrogram of the signal.

    :param signal: signal to compute spectrogram from
    :param window_size: number of samples per FFT
    :param overlap: overlap of each window
    :param power: if power=True, the power spectrum is returned
    :returns: the computed spectrogram
    """
    stp = int(window_size * (1 - overlap))
    stft = np.abs(librosa.stft(y=signal, n_fft=window_size, hop_length=stp))
    if power:
        stft=librosa.amplitude_to_db(stft, ref=np.max)
    return stft


def normalize_stft(stft):
    """
    Convert a spectrogram to units of standard deviation by computing the mean and standard deviation
    over all values in the spectrogram, then subtracting the mean and dividing by the standard deviation
    for each value.

    :param stft: spectrogram to normalize
    :return normalized: the normalized spectrogram
    """
    mean = np.mean(stft)
    std = np.std(stft)
    normalized = np.subtract(stft, mean)
    normalized = np.divide(normalized, std)
    return normalized


def get_subset(stft, rng, fs=32000):
    """
    Get the subset of spectrogram frequency bins that contain frequencies within the given range.

    :param stft: spectrogram to subset
    :param rng: frequency range
    :param fs: sample rate of the signal from which the stft was computed
    :returns: the subsetted spectrogram
    """
    freq_range = fs/2
    hz_per_bin = freq_range / len(stft)
    lo_bin = math.floor(rng[0] / hz_per_bin)
    hi_bin = math.ceil(rng[1] / hz_per_bin)
    subsetted = stft[lo_bin:hi_bin + 1]
    return subsetted


def stft_to_dataframe(stft):
    """
    Converts numpy spectrogram to Pandas DataFrame.

    :param stft: the numpy spectrogram
    :returns: stft as a Pandas DataFrame where indexes are timesteps and columns are frequency bins
    """
    dic = OrderedDict()
    for i in range(len(stft)):
        dic[i] = stft[i]
    return pd.DataFrame.from_dict(dic)


def main(in_dir=conf.wav_path, out_dir=conf.stft_path,
         window_size=conf.window_size, overlap=conf.overlap, subset=conf.subset,
         sigma=conf.sigma, normalize=conf.normalize):

    # Check that input and output directories exist,
    # if not, make them.
    ensure_dir(out_dir)

    # Declare file number and timer variables for
    # keeping track of run time and dataset statistics.
    num_files = 0
    total_song_length = 0
    read_time = 0
    fft_time = 0
    df_conv_time = 0
    pkl_time = 0

    # Preprocess all wav files in the directory specified by in_dir.
    for filename in glob.glob(in_dir + '*.wav'):

        num_files += 1
        print('FILE NAME %s' % filename)
        print('FILE NUMBER %d' % num_files)

        # Read wav and store the filename without it's stem.
        print('Reading file...')
        t_0 = time.time()
        name = filename.split('/')[-1]
        name = name.split('.')[0]
        fs, x = read_wav(filename, verbose=False)
        total_song_length += len(x) / fs
        t_1 = time.time()
        read_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # Compute spectrogram and return the
        # frequency range specified by subset if
        # one has been passed.
        print('Computing spectrogram...')
        t_0 = time.time()
        stft = compute_stft(x, window_size, overlap, power=conf.power)
        if type(subset) == tuple:
            stft = get_subset(stft, subset, fs=fs)
        t_1 = time.time()
        fft_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # Apply Gaussian filter if sigma is specified.
        if sigma is not None:
            print('Gaussian filter with sigma={}...'.format(sigma))
            stft = gaussian_filter(stft, sigma=sigma)
            print('Done.')

        # Normalize the spectrogram if normalize is true.
        if normalize is True:
            print('Normalizing...')
            stft = normalize_stft(stft)
            print('Done.')

        # Convert the spectrogram from ndarray to DataFrame
        # with timestep indexes and frequency bin columns.
        print('Converting to dataframe...')
        t_0 = time.time()
        df = stft_to_dataframe(stft)
        t_1 = time.time()
        df_conv_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # Dump the dataframe as a Pickle to the directory
        # specified by out_dir.
        print('Pickling...')
        t_0 = time.time()
        pkl.dump(df, open(out_dir + name + '.pkl', "wb"))
        t_1 = time.time()
        pkl_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

    # Report that preprocessing has been completed and
    # output runtime and dataset statistics.
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


if __name__ == "__main__":
    main()
