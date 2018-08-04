from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import math


def read_wav(filepath, verbose=False):
    if verbose:
        print('Reading from %s' % filepath)
    fs, x = wavfile.read(filepath)
    if verbose:
        print('Done.')
    if verbose:
        print('Sample Rate: %d' % fs)
        print('Recording Length (s): %f' % (len(x)/fs))
        print('Number of Samples: %d' % len(x))
    return fs, x.astype(float)


def compute_stft(signal, window_size, overlap, db=True):
    stp = int(window_size*(1-overlap))
    stft = np.abs(librosa.stft(y = signal, n_fft=window_size, hop_length=stp))
    if db:
        stft = librosa.amplitude_to_db(stft, ref=np.max)
    return stft

def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    normalized = np.divide( np.subtract(arr, mean), std)
    return normalized


def get_subset(stft, rng, freq_range=16000):
    hz_per_bin = freq_range / len(stft)
    lo_bin = math.floor(rng[0] / hz_per_bin)
    hi_bin = math.ceil(rng[1] / hz_per_bin)
    print('LO %d' % lo_bin)
    print('HI %d' % hi_bin)
    return stft[lo_bin:hi_bin + 1]


if __name__ == '__main__':

    wav_file = '/Users/bergamaschi/Documents/HumpbackSong/HBSe_20151207T070326.wav'

    fs, x = read_wav(wav_file, verbose=True)

    fs = 32000
    window_size = 1024
    ovr = 0.5
    hop_len = window_size*(1-ovr)
    subset = (50, 2000)
    hz_per_bin = (fs/2) / (1+window_size/2)


    # Waveform
    fig = plt.figure(figsize=(16, 4))
    display.waveplot(x, sr=fs)
    plt.title('Waveform (30 kHz Sample Rate)')
    plt.show()


    # Spectrogram
    fig = plt.figure(figsize=(16, 4))
    stft = compute_stft(x, window_size, ovr)
    display.specshow(stft, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (1024 Window, 50% Overlap)')
    plt.show()


    # Subsetted Spectrogram
    fig = plt.figure(figsize=(16, 4))
    stft = get_subset(stft, rng=(50, 2000))
    display.specshow(stft, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Spectrogram (50Hz to 2kHz Subset)')
    plt.show()


    # Gaussian Filtering
    fig = plt.figure(figsize=(16, 4))
    blurred = gaussian_filter(stft, sigma=2)
    display.specshow(blurred, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Gaussian Blur (sigma=2)')
    plt.show()


    # Normalization
    fig = plt.figure(figsize=(16, 4))
    normal = normalize(blurred)
    display.specshow(normal, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f Standard Deviations')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Normalized')
    plt.tight_layout()
    plt.show()


