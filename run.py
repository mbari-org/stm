import stft
import cluster
import labels_to_documents

# TODO: change file types from pickle to JSON


def preprocessing():

    in_dir = '/Users/bergamaschi/Documents/HumpbackSong/test/'

    # stft parameters
    window_size = [1024]
    overlap = [0.5]
    subset = (50, 2000)

    # kmeans parameters
    k_vals = [10000]

    for win in window_size:
        for ovr in overlap:
            stft_dir = stft.run(in_dir, window_size=win, overlap=ovr, subset=subset)
            for k in k_vals:
                cluster.run(stft_dir, k)


if __name__ == '__main__':
    preprocessing()


