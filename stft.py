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
    Read a .wav file and return the signal and sample rate.
    :param fpath: file path of .wav file
    :param verbose: if true, read_wav prints information about the .wav
    :return fs: sample rate of the .wav
    :return signal: signal read from the .wav
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


def compute_stft(signal, window_size, overlap):
    """
    Compute the spectrogram of the signal.
    :param signal: signal to compute spectrogram from
    :param window_size: number of samples per FFT
    :param overlap: overlap of each window
    :return stft: the computed spectrogram
    """
    stp = int(window_size * (1 - overlap))
    # divide by window_size to convert to amplitude
    stft = np.divide(np.abs(librosa.stft(y=signal, n_fft=window_size, hop_length=stp)),
                     window_size)
    return stft


def normalize(stft):
    """
    Convert a spectrogram to units of standard deviation by computing the mean and standard deviation
    over all values in the spectrogram, then subtracting the mean and dividing by the standard deviation
    for each value.
    :param stft: spectrogram to normalize
    :return normalized: the normalized spectrogram
    """
    mean = np.mean(stft.values)
    std = np.std(stft.values)
    normalized = stft.subtract(mean)
    normalized = normalized.divide(std)
    return normalized


def get_subset(stft, rng, fs=32000):
    """
    Get the subset of spectrogram frequency bins that contain frequencies within the given range.
    :param stft: spectrogram to subset
    :param rng: frequency range
    :param fs: sample rate of the signal from which the stft was computed
    :return subsetted: the subsetted spectrogram
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
    :param stft:
    :return:
    """
    dic = OrderedDict()
    for i in range(len(stft)):
        dic[i] = stft[i]
    return pd.DataFrame.from_dict(dic)


def main(in_dir=conf.wav_path, out_dir=conf.stft_path,
         window_size=conf.window_size, overlap=conf.overlap, subset=conf.subset):

    # input and output directories
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
            stft = get_subset(stft, subset, fs=fs)
        t_1 = time.time()
        fft_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # TODO: add gaussian filter timer
        # gaussian filter
        print('Gaussian filtering...')
        stft = gaussian_filter(stft, sigma=2)
        print('Done.')

        # convert to dataframe
        print('Converting to dataframe...')
        t_0 = time.time()
        df = stft_to_dataframe(stft)
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
