import stft
import cluster
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":

    in_dir = "/Users/bergamaschi/Documents/HumpbackSong/dataset/"
    stft.main(in_dir=in_dir)

    k_vals = np.logspace(1, 4, 10, base=10, dtype=int)

    inertia_scores = []

    for k in k_vals:
        inertia = cluster.main(k=k, return_inertia=True)
        inertia_scores.append(inertia)

    fig, ax = plt.subplots(figsize=(5, 5))

    print(k_vals)
    print(inertia_scores)

    plt.plot(k_vals, inertia_scores, marker="x")
    plt.title("Inertia vs. K Value")
    plt.xlabel("K Value")
    plt.ylabel("Inertia")
    plt.show()
