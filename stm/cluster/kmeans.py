# stm, Apache-2.0 license
# Filename: kmeans.py
# Description: K-means clustering using knee method
from __future__ import annotations

import numpy as np

UMAP_COMPONENTS = 15
HDBSCAN_MIN_CLUSTER_SIZE = 5
# K-Means knee sweep
KMEANS_SWEEP_SIZE = 20      # log-spaced K candidates to evaluate
KMEANS_K_MAX = 50           # ceiling on the sweep grid
KMEANS_SWEEP_CAP = 20_000   # subsample this many rows for the sweep itself


def kmeans_k_grid(
    n_samples: int,
    sweep_size: int = KMEANS_SWEEP_SIZE,
    k_max: int = KMEANS_K_MAX,
) -> np.ndarray:
    """Log-spaced K candidates in ``[2, min(n_samples, k_max)]``."""
    n_samples = int(n_samples)
    ceiling = min(n_samples, int(k_max))
    if ceiling < 2:
        raise ValueError(f"K-Means needs at least 2 samples, got {n_samples}")
    if ceiling == 2:
        return np.array([2], dtype=int)
    raw = np.logspace(np.log10(2.0), np.log10(float(ceiling)), num=int(sweep_size))
    return np.unique(np.clip(np.rint(raw), 2, ceiling).astype(int))


def knee_n_clusters(ks: np.ndarray, inertias: np.ndarray) -> int:
    """Kneedle (Satopaa et al. 2011) on a convex decreasing inertia curve.

    Both axes are min-max normalised, then the point furthest below the
    straight line joining the endpoints is taken as the knee.
    """
    ks = np.asarray(ks, dtype=np.int64)
    ys = np.asarray(inertias, dtype=np.float64)
    if ks.ndim != 1 or ys.ndim != 1 or len(ks) != len(ys):
        raise ValueError("ks and inertias must be 1-D arrays of the same length")
    if len(ks) == 0:
        raise ValueError("ks is empty")
    order = np.argsort(ks)
    ks, ys = ks[order], ys[order]
    if len(ks) == 1 or float(np.ptp(ys)) <= 0.0:
        return int(ks[0])
    xn = (ks - ks.min()) / (ks.max() - ks.min())
    yn = (ys - ys.min()) / (ys.max() - ys.min())
    return int(ks[int(np.argmax((1.0 - yn) - xn))])


class KMeansCluster:
    """K-Means codebook whose K is chosen from the inertia-curve knee.

    Takes no K: a log-spaced grid of candidates is swept, the inertia curve
    is recorded, and :func:`knee_n_clusters` picks the elbow. Intended for
    PCEN frames, where the mel vectors form a natural codebook and the frame
    count (tens of thousands) makes UMAP + HDBSCAN impractical.

    The interface matches :class:`Cluster` — ``labels`` and ``n_clusters`` —
    so either can supply discrete word ids to
    :class:`~stm.topicmodel.runner.TopicModelRunner`. Unlike
    :class:`Cluster` there is no noise label; every row gets a codebook id.

    ``k_grid`` and ``inertias`` are kept so the sweep can be plotted.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        random_state: int = 0,
        sweep_size: int = KMEANS_SWEEP_SIZE,
        k_max: int = KMEANS_K_MAX,
        sweep_cap: int = KMEANS_SWEEP_CAP,
    ):
        from sklearn.cluster import KMeans

        self.embeddings = np.asarray(embeddings)
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (n_samples, n_features), got {self.embeddings.shape}"
            )
        n_samples = self.embeddings.shape[0]
        if n_samples < 2:
            raise ValueError(f"need at least 2 rows to cluster, got {n_samples}")

        self.random_state = int(random_state)
        self.k_grid = kmeans_k_grid(n_samples, sweep_size=sweep_size, k_max=k_max)

        sweep = self.embeddings
        if n_samples > int(sweep_cap):
            rng = np.random.RandomState(self.random_state)
            sweep = self.embeddings[
                rng.choice(n_samples, int(sweep_cap), replace=False)
            ]
            print(f"Knee sweep on {len(sweep)} / {n_samples} rows")
        print(f"K-Means knee sweep K={self.k_grid.tolist()}")

        inertias: list[float] = []
        for k in self.k_grid:
            model = KMeans(
                n_clusters=int(k), random_state=self.random_state, n_init="auto"
            ).fit(sweep)
            inertias.append(float(model.inertia_))
            print(f"  K={int(k)} inertia={model.inertia_:.6g}")
        self.inertias = np.asarray(inertias, dtype=np.float64)

        self.k = int(min(knee_n_clusters(self.k_grid, self.inertias), n_samples))
        print(f"Knee K={self.k}; fitting K-Means {self.embeddings.shape}")
        self.kmeans = KMeans(
            n_clusters=self.k, random_state=self.random_state, n_init="auto"
        ).fit(self.embeddings)
        self.labels = np.asarray(self.kmeans.labels_, dtype=np.int64)
        print(f"Found {self.n_clusters} clusters (0 noise)")

    @property
    def n_clusters(self) -> int:
        return int(len(set(self.labels.tolist()) - {-1}))

    def inertia_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """``(k_grid, inertias)`` from the sweep, for plotting the elbow."""
        return self.k_grid, self.inertias


