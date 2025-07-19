"""
This module computes spectrograms from given wav files that
are preprocessed in the manner specified by conf.py
"""
import sklearn
from scipy.io import wavfile
import math
import pandas as pd
import pickle as pkl
import librosa
from collections import OrderedDict
import time
import numpy as np
from scipy.ndimage import gaussian_filter

import conf


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

def compute_stft_pcen(signal, window_size, overlap, fs, fmin, fmax, gain=0.98, bias=2, tc=0.4):
    """
    Compute the PCEN (Per-Channel Energy Normalization) spectrogram of the signal.
    :param signal:  signal to compute PCEN from
    :param window_size:  number of samples per FFT
    :param overlap:  overlap of each window
    :param fs:  sample rate of the signal
    :param fmin:  minimum frequency for the mel filterbank
    :param fmax:  maximum frequency for the mel filterbank
    :param gain:  gain parameter for PCEN
    :param bias:  bias parameter for PCEN
    :param tc:  time constant for PCEN, in seconds
    :return: the PCEN spectrogram of the signal
    """
    hop_length = int(window_size * (1 - overlap))
    stft_mel = librosa.feature.melspectrogram(
        y=sklearn.preprocessing.minmax_scale(signal, feature_range=((-2 ** 31), (2 ** 31))),
        sr=fs,
        fmin=fmin,
        fmax=fmax,
        n_fft=window_size,
        n_mels=256,  # Number of mel bands
        hop_length=hop_length)

    pcen_s = librosa.pcen(stft_mel * (2 ** 31), sr=fs, hop_length=hop_length, gain=gain, bias=bias,
                          time_constant=tc)
    pcen_s = librosa.power_to_db(pcen_s, ref=np.max)
    return pcen_s

def compute_stft(signal, window_size, overlap):
    """
    Compute the spectrogram of the signal.

    :param signal: signal to compute spectrogram from
    :param window_size: number of samples per FFT
    :param overlap: overlap of each window
    :returns: the computed spectrogram
    """
    stp = int(window_size * (1 - overlap))
    stft = np.abs(librosa.stft(y=signal, n_fft=window_size, hop_length=stp))
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
    :returns: the subset spectrogram
    """
    freq_range = fs/2
    hz_per_bin = freq_range / len(stft)
    lo_bin = math.floor(rng[0] / hz_per_bin)
    hi_bin = math.ceil(rng[1] / hz_per_bin)
    subsetted = stft[lo_bin:hi_bin + 1]
    return subsetted


def stft_to_dataframe(stft, times, frequencies):
    """
    Converts numpy spectrogram to Pandas DataFrame.

    :param stft: the numpy spectrogram
    :param times: the time bins of the spectrogram
    :param frequencies: the frequency bins of the spectrogram
    :returns: stft as a Pandas DataFrame where indexes are timesteps and columns are frequency bins
    """
    frequencies = frequencies.squeeze()
    times = times.squeeze()
    df = pd.DataFrame(stft, index=frequencies, columns=times)
    return df


def main(in_dir=conf.wav_path, out_dir=conf.stft_path,
         window_size=conf.window_size, overlap=conf.overlap, subset=conf.subset,
         sigma=conf.sigma, normalize=conf.normalize, use_pcen=conf.use_pcen,
         pcen_gain=conf.pcen_gain, pcen_bias=conf.pcen_bias, pcen_time_constant=conf.pcen_time_constant):

    # Check that input and output directories exist,
    # if not, make them.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Declare file number and timer variables for
    # keeping track of run time and dataset statistics.
    num_files = 0
    total_song_length = 0
    read_time = 0
    fft_time = 0
    df_conv_time = 0
    pkl_time = 0

    # Preprocess all wav files in the directory specified by in_dir.
    for filename in sorted(in_dir.rglob('*.wav')):

        num_files += 1
        print('FILE NAME %s' % filename)
        print('FILE NUMBER %d' % num_files)

        # Read wav and store the filename without its stem.
        print('Reading file...')
        t_0 = time.time()
        name = filename.stem
        fs, x = read_wav(filename, verbose=False)
        total_song_length += len(x) / fs
        t_1 = time.time()
        read_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # Compute the frequencies for the spectrogram. These are used later in visualization.
        frequencies = np.fft.rfftfreq(window_size, d=1.0 / fs)

        # Compute spectrogram and return the
        # frequency range specified by subset if
        # one has been passed.
        print('Computing spectrogram...')
        t_0 = time.time()
        if use_pcen:
            print('Computing PCEN...')
            if type(subset) == tuple:
                fmin, fmax = subset
            else:
                fmin, fmax = 0, fs / 2

            stft = compute_stft_pcen(x, window_size, overlap, fs, fmin, fmax, gain=pcen_gain, bias=pcen_bias, tc=pcen_time_constant)
            # Get the mel filterbank frequencies
            frequencies = librosa.mel_frequencies(n_mels=stft.shape[0], fmin=fmin, fmax=fmax)
        else:
            print('Computing STFT...')
            stft = compute_stft(x, window_size, overlap)

            if type(subset) == tuple:
                # Get the subset of the spectrogram based on the frequencies
                start_idx = np.searchsorted(frequencies, subset[0], side="left")
                end_idx = np.searchsorted(frequencies, subset[1], side="right")
                stft = stft[start_idx:end_idx]
                frequencies = frequencies[start_idx:end_idx]

        times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=fs, hop_length=int(window_size * (1 - overlap)))

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
        df = stft_to_dataframe(stft, times, frequencies)
        t_1 = time.time()
        df_conv_time += (t_1 - t_0)
        print('Done. (%f seconds)' % (t_1 - t_0))

        # Dump the dataframe as a Pickle to the directory
        # specified by out_dir.
        print('Pickling...')
        t_0 = time.time()
        pkl.dump(df, open(out_dir / f'{name}.pkl', "wb"))
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
