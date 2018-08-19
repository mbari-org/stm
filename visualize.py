from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle as pkl

import conf

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

def topic_bar(model_path=conf.model_path, stft_path=conf.stft_path, target_file=conf.target_file,
              window_size=conf.window_size, overlap=conf.overlap, fs=conf.sample_rate):

    theta = pd.read_csv(model_path + "theta.csv", header=None).values

    start_t = 17
    end_t = 66

    words_per_doc = 32
    secs_per_doc = secs_per_frame * words_per_doc
    start_doc = math.floor(start_t / secs_per_doc)
    end_doc = math.ceil(end_t / secs_per_doc)

    real_start_t = secs_per_doc * start_doc
    real_end_t = secs_per_doc * end_doc

    theta_sub = theta[start_doc: end_doc]

    # spectrogram params
    fs = 32000
    window_size = 1024
    overlap = 0.5
    hop_len = window_size * (1 - overlap)
    secs_per_frame = window_size * (1 - overlap) / fs

    # subset of theta
    theta_sub = theta[start_doc: end_doc]

    # barplot params
    N = len(theta_sub)
    P = []
    ind = np.arange(N)
    bottom = np.zeros(len(theta_sub))
    width = 1

    fig = plt.figure(figsize=(16, 8))

    start_frame = int(real_start_t * (1 / secs_per_frame))
    end_frame = int(real_end_t * (1 / secs_per_frame))

    plt.subplot(2, 1, 1)
    Sxx = pkl.load(open(stft_path+target_file+'.pkl', "rb"))
    Sxx_sub = Sxx[start_frame:end_frame]
    Sxx_sub = flip_list(Sxx_sub.values)

    display.specshow(librosa.amplitude_to_db(Sxx_sub, ref=np.max),
                     y_axis='log', x_axis='time',
                     sr=32000, hop_length=512)

    plt.subplot(2, 1, 2)
    for z in range(T):
        P.append(plt.bar(x=ind, height=theta_sub[:, z], width=width, bottom=bottom))
        bottom = np.add(theta_sub[:, z], bottom)
    plt.xlim(xmin=0, xmax=len(theta_sub))
    plt.legend(tuple(_[0] for _ in P), ['Topic %d' % (i) for i in range(T)], loc='lower right')
    plt.xlabel("Documents")
    plt.ylabel("Topic Probability")

    plt.tight_layout()
    plt.show()


def visualize_preproc():
    wav_file = '/Users/bergamaschi/Documents/HumpbackSong/HBSe_20151207T070326.wav'

    fs, x = read_wav(wav_file, verbose=True)

    fs = 32000
    window_size = 1024
    ovr = 0.5
    hop_len = window_size*(1-ovr)
    subset = (50, 2000)
    hz_per_bin = (fs/2) / (1+window_size/2)


    # Waveform
    fig = plt.figure(figsize=(12, 4))
    display.waveplot(x, sr=fs)
    plt.title('Waveform (30 kHz Sample Rate)')
    plt.tight_layout()
    plt.show()


    # Spectrogram
    fig = plt.figure(figsize=(12, 4))
    stft = compute_stft(x, window_size, ovr)
    display.specshow(stft, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (1024 Window, 50% Overlap)')
    plt.tight_layout()
    plt.show()


    # Subsetted Spectrogram
    fig = plt.figure(figsize=(12, 4))
    stft = get_subset(stft, rng=(50, 2000))
    display.specshow(stft, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Spectrogram (50Hz to 2kHz Subset)')
    plt.tight_layout()
    plt.show()


    # Gaussian Filtering
    fig = plt.figure(figsize=(12, 4))
    blurred = gaussian_filter(stft, sigma=2)
    display.specshow(blurred, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Gaussian Filter (sigma=2)')
    plt.tight_layout()
    plt.show()


    # Normalization
    fig = plt.figure(figsize=(12, 4))
    normal = normalize(blurred)
    display.specshow(normal, y_axis='linear', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f Standard Deviations')
    locs, labels = plt.yticks()
    new_labels = ["%.2f"%(hz_per_bin+((locs[i]/16000)*len(stft)*hz_per_bin)) for i in range(len(locs))]
    plt.yticks(locs, new_labels)
    plt.title('Normalized')
    plt.tight_layout()
    plt.show()
