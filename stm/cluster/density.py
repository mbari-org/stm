# stm, Apache-2.0 license
# Filename: cluster/density.py
# Description: HDBSCAN clustering of UMAP-reduced embeddings
from __future__ import annotations

import shutil
from pathlib import Path

import hdbscan
import numpy as np
import onnxruntime as ort
import soundfile as sf
import umap

from stm.config import Config
from stm.embed import PERCH_INPUT_SAMPLES, _ensure_onnx_model, _fill_to_length, peak_normalize

UMAP_COMPONENTS = 10
UMAP_NEIGHBORS = 15
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES: int | None = None  # None -> same as min_cluster_size
HDBSCAN_ALLOW_SINGLE_CLUSTER = False


def density_params(config: Config | None) -> dict:
    """UMAP/HDBSCAN keyword arguments for :class:`DensityCluster` from *config*."""
    if config is None:
        return {}
    return {
        "n_components": config.umap_components,
        "n_neighbors": config.umap_neighbors,
        "min_cluster_size": config.hdbscan_min_cluster_size,
        "min_samples": config.hdbscan_min_samples,
        "allow_single_cluster": config.hdbscan_allow_single_cluster,
    }

class DensityCluster:
    """Cluster embeddings with cosine UMAP (``n_components``-D) then HDBSCAN.

    ``min_samples`` defaults to ``min_cluster_size``. Both are capped at the
    number of embeddings so small inputs do not raise.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        random_state: int = 0,
        assign_noise: bool = False,
        *,
        n_components: int = UMAP_COMPONENTS,
        n_neighbors: int = UMAP_NEIGHBORS,
        min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples: int | None = HDBSCAN_MIN_SAMPLES,
        allow_single_cluster: bool = HDBSCAN_ALLOW_SINGLE_CLUSTER,
    ):
        self.embeddings = np.asarray(embeddings)
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (n_samples, n_features), got {self.embeddings.shape}"
            )
        n_samples = self.embeddings.shape[0]
        if n_samples < 2:
            raise ValueError(f"need at least 2 embeddings to cluster, got {n_samples}")

        n_neighbors = min(int(n_neighbors), n_samples - 1)
        n_components = min(int(n_components), n_samples - 1)
        print(
            f"UMAP {self.embeddings.shape} -> {n_components}-D "
            f"(cosine, n_neighbors={n_neighbors})"
        )
        self.umap = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            random_state=random_state,
            n_jobs=1,
        )
        self.reduced = self.umap.fit_transform(self.embeddings)

        mcs = max(2, min(int(min_cluster_size), n_samples))
        ms = mcs if min_samples is None else int(min_samples)
        ms = max(1, min(ms, n_samples - 1))
        if mcs != int(min_cluster_size):
            print(f"min_cluster_size {min_cluster_size} capped to {mcs} for {n_samples} samples")
        print(
            f"HDBSCAN min_cluster_size={mcs} min_samples={ms} "
            f"allow_single_cluster={allow_single_cluster}"
        )
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            allow_single_cluster=bool(allow_single_cluster),
        )
        self.labels = self.hdbscan.fit_predict(self.reduced)
        n_noise = int(np.sum(self.labels == -1))
        if assign_noise and n_noise:
            self.labels = self._assign_noise_to_nearest()
            n_left = int(np.sum(self.labels == -1))
            print(
                f"Found {self.n_clusters} clusters "
                f"(assigned {n_noise - n_left} noise to nearest cluster)"
            )
        else:
            print(f"Found {self.n_clusters} clusters ({n_noise} noise)")

    @property
    def n_clusters(self) -> int:
        return int(len(set(self.labels.tolist()) - {-1}))

    def _assign_noise_to_nearest(self) -> np.ndarray:
        """Replace HDBSCAN noise (``-1``) with the nearest cluster centroid."""
        points = self.reduced
        labels = np.asarray(self.labels).copy()
        noise = labels == -1
        if not np.any(noise):
            return labels
        clustered = ~noise
        if not np.any(clustered):
            print("No clusters to assign noise to; leaving labels as -1")
            return labels

        ids = np.unique(labels[clustered])
        centroids = np.stack([points[labels == k].mean(axis=0) for k in ids])
        dist = np.linalg.norm(points[noise][:, None, :] - centroids[None, :, :], axis=2)
        labels[noise] = ids[dist.argmin(axis=1)]
        return labels


def _wavs_in_directory(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def _consolidate_wavs(directory: Path) -> int:
    """Move WAV files from subdirectories up into *directory* and drop empty dirs."""
    n_moved = 0
    for wav in sorted(directory.rglob("*.wav")):
        if wav.parent == directory:
            continue
        dest = directory / wav.name
        if dest.exists() and dest.resolve() != wav.resolve():
            dest.unlink()
        shutil.move(wav.as_posix(), dest.as_posix())
        n_moved += 1

    for sub in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
        if sub == directory:
            continue
        try:
            sub.rmdir()
        except OSError:
            pass
    return n_moved


def _embed_wavs(
    clips: list[Path],
    model_path: Path,
    perch_audio_seconds: float,
    perch_window_fill: str,
    batch_size: int = 32,
) -> np.ndarray:
    model_path = _ensure_onnx_model(Path(model_path))
    session = ort.InferenceSession(model_path.as_posix())
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for output in session.get_outputs():
        if output.name == "embedding":
            output_name = output.name
            break

    windows = []
    for clip in clips:
        audio, sr = sf.read(clip.as_posix(), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio[:, 0]
        keep = int(round(perch_audio_seconds * sr))
        windows.append(
            _fill_to_length(
                np.asarray(audio[:keep], dtype=np.float32),
                PERCH_INPUT_SAMPLES,
                perch_window_fill,
            )
        )
    batch = peak_normalize(np.stack(windows))
    parts = []
    for lo in range(0, len(batch), batch_size):
        hi = min(lo + batch_size, len(batch))
        parts.append(
            np.asarray(
                session.run([output_name], {input_name: batch[lo:hi]})[0],
                dtype=np.float32,
            )
        )
    return np.concatenate(parts, axis=0)


def cluster_directory(
    directory: Path | str,
    model_path: Path | str,
    config: Config | None = None,
    assign_noise: bool = True,
    random_state: int = 0,
) -> DensityCluster:
    """Cluster WAV clips in *directory* and move them into numbered subfolders.

    Files in ``background/`` become ``background/background_0/``,
    ``background/background_1/``, and so on. Any existing subdirectories are
    flattened into *directory* first so a re-run clusters the full set.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    n_flat = _consolidate_wavs(directory)
    if n_flat:
        print(f"Consolidated {n_flat} clips into {directory}")

    clips = _wavs_in_directory(directory)
    if len(clips) < 2:
        raise ValueError(f"need at least 2 WAV files in {directory}, found {len(clips)}")

    perch_audio_seconds = 1.0 if config is None else config.perch_audio_seconds
    perch_window_fill = "fill" if config is None else config.perch_window_fill

    print(f"Embedding {len(clips)} clips in {directory}")
    embeddings = _embed_wavs(
        clips, Path(model_path), perch_audio_seconds, perch_window_fill
    )
    cluster = DensityCluster(
        embeddings,
        random_state=random_state,
        assign_noise=assign_noise,
        **density_params(config),
    )

    stem = directory.name
    counts: dict[int, int] = {}
    for clip, label in zip(clips, cluster.labels):
        label = int(label)
        dest_dir = directory / f"{stem}_{label}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(clip.as_posix(), (dest_dir / clip.name).as_posix())
        counts[label] = counts.get(label, 0) + 1

    for label in sorted(counts):
        print(f"Moved {counts[label]} clips to {directory / f'{stem}_{label}'}")
    return cluster