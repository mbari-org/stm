import pickle as pkl
import numpy as np
import pandas as pd
import librosa
from librosa import display
import matplotlib.pyplot as plt


import visualize
from stft import read_wav, compute_stft


if __name__ == "__main__":


    data = [[0.1, 0.1, 0.8],
            [0.2, 0.3, 0.5]]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax = visualize.stacked_bar(data)


    fig.show()
