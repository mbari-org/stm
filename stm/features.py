# stm, Apache-2.0 license
# Filename: features.py
# Description: Feature extractors on one or more time grids
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter

from perchtopic.cache import (
    PerchCacheKey,
    atomic_write_json,
    atomic_write_npy,
    default_perch_cache_root,
    perch_cache_stem,
)
from perchtopic.classify import (
    PERCH2_EMBEDDING_DIM,
    PERCH2_SAMPLE_RATE,
    PERCH2_WINDOW_SECONDS,
)
from perchtopic.embed import (
    PERCH_INPUT_SAMPLES,
    _fill_to_length,
    embed_windows,
    load_onnx_session,
)


@dataclass(frozen=True, eq=False)
class TimeGrid:
    """Analysis intervals in seconds.

    ``edges`` is ``(T, 2)`` start/end. Regular hop/window grids are a special
    case of the same representation — build them with :meth:`regular`.
    """

    edges: np.ndarray

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must have shape (T, 2), got {edges.shape}")
        if np.any(edges[:, 1] < edges[:, 0]):
            raise ValueError("each interval must have end >= start")
        edges = np.ascontiguousarray(edges)
        edges.flags.writeable = False
        object.__setattr__(self, "edges", edges)

    def __len__(self) -> int:
        return int(self.edges.shape[0])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeGrid):
            return NotImplemented
        return np.array_equal(self.edges, other.edges)

    @property
    def starts(self) -> np.ndarray:
        return self.edges[:, 0]

    @property
    def ends(self) -> np.ndarray:
        return self.edges[:, 1]

    @property
    def durations(self) -> np.ndarray:
        return self.ends - self.starts

    @classmethod
    def regular(cls, start: float, hop: float, window: float, n: int) -> TimeGrid:
        """``n`` intervals ``[start + i*hop, start + i*hop + window)``."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if hop <= 0:
            raise ValueError(f"hop must be > 0, got {hop}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        starts = start + np.arange(n, dtype=np.float64) * hop
        ends = starts + window
        return cls(np.column_stack([starts, ends]))


@dataclass
class FeatureBlock:
    """Features on one :class:`TimeGrid`. ``values`` is ``(T, D)``."""

    values: np.ndarray
    grid: TimeGrid
    mask: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(f"values must have shape (T, D), got {values.shape}")
        if values.shape[0] != len(self.grid):
            raise ValueError(
                f"values T={values.shape[0]} does not match grid T={len(self.grid)}"
            )
        self.values = values
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != (len(self.grid),):
                raise ValueError(
                    f"mask must have shape ({len(self.grid)},), got {mask.shape}"
                )
            self.mask = mask


def pool(block: FeatureBlock, target: TimeGrid) -> FeatureBlock:
    """Resample embeddings onto *target* by overlap-weighted mean+std.

    Each target row is ``[mean | std]`` of source frames whose intervals
    overlap that target interval. Rows with no overlap (or only masked
    source frames) are zero with ``mask=False``.
    """
    src = block.values.astype(np.float64, copy=False)
    s_edges = block.grid.edges
    t_edges = target.edges
    s, d = src.shape
    t = len(target)
    valid = np.ones(s, dtype=bool) if block.mask is None else np.asarray(block.mask, dtype=bool)

    overlap = np.maximum(
        0.0,
        np.minimum(s_edges[:, 1:2], t_edges[:, 1])
        - np.maximum(s_edges[:, 0:1], t_edges[:, 0]),
    )
    overlap[~valid, :] = 0.0
    weight_sum = overlap.sum(axis=0)
    mask = weight_sum > 0.0
    weights = np.zeros_like(overlap)
    np.divide(overlap, weight_sum, out=weights, where=mask)

    mean = weights.T @ src
    mean_sq = weights.T @ (src * src)
    std = np.sqrt(np.maximum(0.0, mean_sq - mean * mean))
    values = np.concatenate([mean, std], axis=1).astype(np.float32)
    values[~mask] = 0.0
    provenance = dict(block.provenance)
    provenance["pool"] = "mean_std"
    return FeatureBlock(values=values, grid=target, mask=mask, provenance=provenance)


def combine(blocks: Sequence[FeatureBlock], target: TimeGrid) -> FeatureBlock:
    """Pool each block onto *target* (mean+std) and concatenate along D."""
    if not blocks:
        raise ValueError("combine() requires at least one FeatureBlock")
    pooled = [pool(block, target) for block in blocks]
    values = np.concatenate([block.values for block in pooled], axis=1)
    mask = np.logical_and.reduce([block.mask for block in pooled])
    return FeatureBlock(
        values=values,
        grid=target,
        mask=mask,
        provenance={
            "combine": "mean_std",
            "blocks": [block.provenance for block in pooled],
        },
    )


def _wav_paths(source: Path) -> list[Path]:
    source = Path(source)
    if source.is_dir():
        clips = sorted(p for p in source.rglob("*.wav") if p.is_file())
        if not clips:
            raise FileNotFoundError(f"No wav clips under {source}")
        return clips
    if not source.is_file():
        raise FileNotFoundError(source)
    return [source]


def compute_stft_pcen(
    signal,
    window_size,
    overlap,
    fs,
    fmin,
    fmax,
    num_mel_bins=32,
    gain=0.98,
    bias=2,
    tc=0.4,
    power_to_db=True,
):
    """PCEN (Per-Channel Energy Normalization) mel-spectrogram.

    Ported from ``perch2topic/stft.py``. Returns ``(n_mels, n_frames)``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    hop_length = int(window_size * (1 - overlap))
    if hop_length < 1:
        raise ValueError(f"hop_length must be >= 1, got {hop_length}")
    if signal.size == 0:
        return np.zeros((num_mel_bins, 0), dtype=np.float32)

    peak_to_peak = float(np.max(signal) - np.min(signal))
    min_val, max_val = -(2**31), 2**31
    if peak_to_peak == 0.0:
        signal_scaled = np.zeros_like(signal, dtype=np.float64)
    else:
        signal_scaled = (signal - np.min(signal)) / peak_to_peak
        signal_scaled = signal_scaled * (max_val - min_val) + min_val

    stft_mel = librosa.feature.melspectrogram(
        y=signal_scaled,
        sr=fs,
        fmin=fmin,
        fmax=fmax,
        n_fft=window_size,
        n_mels=num_mel_bins,
        hop_length=hop_length,
    )
    pcen_s = librosa.pcen(
        stft_mel * (2**31),
        sr=fs,
        hop_length=hop_length,
        gain=gain,
        bias=bias,
        time_constant=tc,
    )
    if power_to_db:
        pcen_s = librosa.power_to_db(pcen_s, ref=np.max)
    return np.asarray(pcen_s, dtype=np.float32)


def _regular_grid_params(grid: TimeGrid) -> dict[str, float | int] | None:
    """Hop/window/start/n when *grid* is a regular :meth:`TimeGrid.regular` grid."""
    n = len(grid)
    if n == 0:
        return {"start": 0.0, "hop": 0.0, "window": 0.0, "n": 0}
    windows = grid.durations
    starts = grid.starts
    if n == 1:
        window = float(windows[0])
        return {"start": float(starts[0]), "hop": window, "window": window, "n": 1}
    hops = np.diff(starts)
    if not (np.allclose(hops, hops[0]) and np.allclose(windows, windows[0])):
        return None
    return {
        "start": float(starts[0]),
        "hop": float(hops[0]),
        "window": float(windows[0]),
        "n": int(n),
    }


def _concat_file_blocks(blocks: Sequence[FeatureBlock], provenance: dict) -> FeatureBlock:
    if len(blocks) == 1:
        return blocks[0]
    values = np.concatenate([block.values for block in blocks], axis=0)
    mask = np.concatenate(
        [
            np.ones(len(block.grid), dtype=bool) if block.mask is None else block.mask
            for block in blocks
        ],
        axis=0,
    )
    edges = np.concatenate([block.grid.edges for block in blocks], axis=0)
    return FeatureBlock(
        values=values,
        grid=TimeGrid(edges),
        mask=mask,
        provenance=provenance,
    )


class Perch2Extractor:
    """Perch v2 ONNX embeddings on a :class:`TimeGrid`.

    Native window is 5 s at 32 kHz. ``timeout`` is the wall-clock budget in
    seconds for one :meth:`extract` call; ``None`` means no limit.

    *source* may be a WAV file or a directory; directories are expanded with
    ``rglob('*.wav')`` and each file is extracted on the same grid, then
    concatenated along T.

    Window embeddings are cached under ``output_path/perch_raw/<hop>_<audio>_<fill>/``
    (or ``<dataset>/perch_raw/`` when *output_path* is omitted) so a later
    :meth:`extract` — or a KeyboardInterrupt restart — skips ONNX for windows
    already written.
    """
    import os
    sample_rate = PERCH2_SAMPLE_RATE
    window_seconds = PERCH2_WINDOW_SECONDS
    embedding_dim = PERCH2_EMBEDDING_DIM
    this_dir = os.path.dirname(__file__)

    def __init__(
        self,
        model: Path | None = None,
        timeout: float | None = None,
        window_fill: str = "pad",
        batch_size: int = 32,
        output_path: Path | None = None,
        cache: bool = True,
    ) -> None:
        self.model = None if model is None else Path(model)
        if self.model:
            if not self.model.exists():
                raise ValueError(f"Perch2Extractor model:{model} does not exist")
            if not self.model.is_file():
                raise ValueError(f"Perch2Extractor model:{model} must be a file not a directory")
        else:
            from perchtopic.embed import PERCH_ONNX_URL, _ensure_onnx_model
            self.model = _ensure_onnx_model(Path(PERCH_ONNX_URL))
        self.timeout = timeout
        self.window_fill = window_fill
        self.batch_size = batch_size
        self.output_path = None if output_path is None else Path(output_path)
        self.cache = bool(cache)
        self._session = None

    @classmethod
    def from_config(
        cls,
        config,
        model: Path | None = None,
        timeout: float | None = None,
        batch_size: int = 32,
        cache: bool = True,
    ) -> "Perch2Extractor":
        """Build from :class:`~stm.config.Config` Perch window settings."""
        return cls(
            model=model,
            timeout=timeout,
            window_fill=config.perch_window_fill,
            batch_size=batch_size,
            output_path=config.output_path,
            cache=cache,
        )

    def feature_width(self) -> int:
        """Embedding vector length (``D`` in ``values`` shape ``(T, D)``)."""
        return int(self.embedding_dim)

    def _onnx_session(self):
        if self.model is None:
            raise ValueError("Perch2Extractor requires model")
        if self._session is None:
            self._session = load_onnx_session(self.model)
        return self._session

    def extract(self, source: Path, grid: TimeGrid) -> FeatureBlock:
        """Embed *source*, a WAV file or a directory of WAVs (``rglob('*.wav')``)."""
        if self.timeout is None:
            return self._extract(source, grid)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._extract, source, grid)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeout as exc:
                raise TimeoutError(
                    f"Perch2Extractor.extract exceeded {self.timeout}s"
                ) from exc

    def _extract(self, source: Path, grid: TimeGrid) -> FeatureBlock:
        source = Path(source)
        wavs = _wav_paths(source)
        blocks = [self._extract_file(wav, grid, dataset=source) for wav in wavs]
        provenance = {
            "extractor": "Perch2Extractor",
            "source": str(source),
            "sources": [block.provenance["source"] for block in blocks],
            "model": str(self.model),
            "window_fill": self.window_fill,
            "batch_size": self.batch_size,
        }
        cache_dirs = [block.provenance.get("cache_dir") for block in blocks]
        if any(cache_dirs):
            provenance["cache_dir"] = next(d for d in cache_dirs if d)
        return _concat_file_blocks(blocks, provenance=provenance)

    def _cache_root(self, source: Path) -> Path:
        if self.output_path is not None:
            return Path(self.output_path)
        return default_perch_cache_root(source)

    def _cache_key(self, wav: Path, source: Path, grid: TimeGrid) -> PerchCacheKey:
        stat = wav.stat()
        params = _regular_grid_params(grid)
        model = str(Path(self.model).resolve()) if self.model is not None else ""
        if params is None:
            digest = hashlib.sha1(np.ascontiguousarray(grid.edges).tobytes()).hexdigest()[:16]
            start = float(grid.starts[0]) if len(grid) else 0.0
            return PerchCacheKey(
                wav_path=wav.resolve(),
                output_path=self._cache_root(source),
                window_fill=self.window_fill,
                model=model,
                grid_start=start,
                grid_hop=None,
                grid_window=None,
                grid_n=len(grid),
                wav_mtime_ns=int(stat.st_mtime_ns),
                wav_size=int(stat.st_size),
                edges_sha1=digest,
            )
        return PerchCacheKey(
            wav_path=wav.resolve(),
            output_path=self._cache_root(source),
            window_fill=self.window_fill,
            model=model,
            grid_start=float(params["start"]),
            grid_hop=float(params["hop"]),
            grid_window=float(params["window"]),
            grid_n=int(params["n"]),
            wav_mtime_ns=int(stat.st_mtime_ns),
            wav_size=int(stat.st_size),
        )

    def _cache_paths(self, key: PerchCacheKey, wav: Path, source: Path) -> tuple[Path, Path, Path]:
        stem = perch_cache_stem(wav, source)
        cache_dir = key.cache_dir()
        return (
            cache_dir / f"{stem}.values.npy",
            cache_dir / f"{stem}.mask.npy",
            cache_dir / f"{stem}.metadata.json",
        )

    def _load_file_cache(
        self, key: PerchCacheKey, wav: Path, source: Path
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        values_path, mask_path, meta_path = self._cache_paths(key, wav, source)
        if not (values_path.exists() and mask_path.exists() and meta_path.exists()):
            return None
        try:
            stored = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not key.matches(stored):
            return None
        values = np.load(values_path)
        mask = np.load(mask_path)
        n_completed = int(stored.get("n_completed", 0))
        if values.shape != (key.grid_n, PERCH2_EMBEDDING_DIM):
            return None
        if mask.shape != (key.grid_n,):
            return None
        return values.astype(np.float32, copy=False), np.asarray(mask, dtype=bool), n_completed

    def _write_file_cache(
        self,
        key: PerchCacheKey,
        wav: Path,
        source: Path,
        values: np.ndarray,
        mask: np.ndarray,
        n_completed: int,
        n_valid: int,
    ) -> None:
        values_path, mask_path, meta_path = self._cache_paths(key, wav, source)
        atomic_write_npy(values_path, np.asarray(values, dtype=np.float32))
        atomic_write_npy(mask_path, np.asarray(mask, dtype=bool))
        payload = key.as_dict()
        payload["n_completed"] = int(n_completed)
        payload["n_valid"] = int(n_valid)
        payload["embedding_dim"] = PERCH2_EMBEDDING_DIM
        atomic_write_json(meta_path, payload)

    def _file_provenance(self, source: Path, cache_dir: Path | None = None) -> dict:
        provenance = {
            "extractor": "Perch2Extractor",
            "source": str(source),
            "model": str(self.model),
            "window_fill": self.window_fill,
            "batch_size": self.batch_size,
        }
        if cache_dir is not None:
            provenance["cache_dir"] = str(cache_dir)
        return provenance

    def _valid_window_index(
        self, grid: TimeGrid, sr: int, n_frames: int
    ) -> tuple[np.ndarray, list[int]]:
        t = len(grid)
        mask = np.zeros(t, dtype=bool)
        valid_idx: list[int] = []
        for i, (t0, t1) in enumerate(grid.edges):
            lo = int(round(float(t0) * sr))
            hi = int(round(float(t1) * sr))
            lo = max(0, min(lo, n_frames))
            hi = max(0, min(hi, n_frames))
            if hi <= lo:
                continue
            valid_idx.append(i)
            mask[i] = True
        return mask, valid_idx

    def _read_filled_window(self, source: Path, grid: TimeGrid, index: int, sr: int, n_frames: int) -> np.ndarray:
        t0, t1 = grid.edges[index]
        lo = int(round(float(t0) * sr))
        hi = int(round(float(t1) * sr))
        lo = max(0, min(lo, n_frames))
        hi = max(0, min(hi, n_frames))
        segment, _ = sf.read(source.as_posix(), start=lo, stop=hi, always_2d=False)
        if getattr(segment, "ndim", 1) > 1:
            segment = segment[:, 0]
        return _fill_to_length(
            np.asarray(segment, dtype=np.float32),
            PERCH_INPUT_SAMPLES,
            self.window_fill,
        )

    def _extract_file(
        self, source: Path, grid: TimeGrid, dataset: Path | None = None
    ) -> FeatureBlock:
        dataset = Path(source if dataset is None else dataset)
        info = sf.info(source.as_posix())
        sr = info.samplerate
        n_frames = info.frames
        t = len(grid)
        mask, valid_idx = self._valid_window_index(grid, sr, n_frames)
        n_valid = len(valid_idx)
        values = np.zeros((t, PERCH2_EMBEDDING_DIM), dtype=np.float32)

        key: PerchCacheKey | None = None
        n_done = 0
        if self.cache:
            key = self._cache_key(source, dataset, grid)
            cached = self._load_file_cache(key, source, dataset)
            if cached is not None:
                cached_values, cached_mask, n_completed = cached
                if n_completed >= n_valid and np.array_equal(cached_mask, mask):
                    print(f"Using cached embeddings {cached_values.shape} in {key.cache_dir()}")
                    return FeatureBlock(
                        values=cached_values,
                        grid=grid,
                        mask=cached_mask,
                        provenance=self._file_provenance(source, key.cache_dir()),
                    )
                if np.array_equal(cached_mask, mask):
                    values = cached_values
                    n_done = min(n_completed, n_valid)
                    if n_done:
                        print(
                            f"Resuming {source} from window {n_done + 1}/{n_valid} "
                            f"in {key.cache_dir()}"
                        )

        if n_valid and n_done < n_valid:
            remain = valid_idx[n_done:]
            print(f"Embedding {n_valid} windows from {source}")
            windows = [
                self._read_filled_window(source, grid, i, sr, n_frames) for i in remain
            ]
            batch = np.stack(windows).astype(np.float32)
            session = self._onnx_session()
            for lo in range(0, len(remain), self.batch_size):
                hi = min(lo + self.batch_size, len(remain))
                embedded = embed_windows(
                    batch[lo:hi],
                    session,
                    batch_size=self.batch_size,
                    progress_offset=n_done + lo,
                    progress_total=n_valid,
                )
                values[remain[lo:hi]] = embedded
                if key is not None:
                    self._write_file_cache(
                        key,
                        source,
                        dataset,
                        values,
                        mask,
                        n_completed=n_done + hi,
                        n_valid=n_valid,
                    )
        elif key is not None:
            self._write_file_cache(key, source, dataset, values, mask, n_done, n_valid)

        return FeatureBlock(
            values=values,
            grid=grid,
            mask=mask,
            provenance=self._file_provenance(
                source, None if key is None else key.cache_dir()
            ),
        )

    def run(
        self,
        source: Path,
        grids: Sequence[TimeGrid],
        target: TimeGrid | None = None,
    ) -> FeatureBlock | list[FeatureBlock]:
        """Extract on each grid; :func:`combine` onto *target* when given."""
        if not grids:
            raise ValueError("run() requires at least one TimeGrid")
        source = Path(source)
        blocks = [self.extract(source, grid) for grid in grids]
        if target is not None:
            return combine(blocks, target)
        if len(blocks) == 1:
            return blocks[0]
        return blocks


class PcenExtractor:
    """PCEN mel-spectrogram **frames** (stm / perch2topic ``stft.py``).

    One FeatureBlock row per STFT hop, not a mean-pooled analysis window.
    Hop is ``window_size * (1 - overlap) / sr`` (8 ms at 512 / 50% / 32 kHz).
    ``grid`` is ignored for resolution; the whole file is extracted so a
    Perch-style 0.5 s grid cannot collapse the spectrogram.

    Defaults match ``perch2topic/conf.py``: gain 0.5, fmin 0, fmax 8000,
    Gaussian ``sigma`` 1.0.

    Frames are emitted as raw PCEN vectors of width ``num_mel_bins``; no
    quantization is applied here.
    """

    def __init__(
        self,
        timeout: float | None = None,
        window_size: int = 512,
        overlap: float = 0.5,
        num_mel_bins: int = 32,
        gain: float = 0.5,
        bias: float = 2.0,
        time_constant: float = 0.4,
        power_to_db: bool = True,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        sigma: float | None = 1.0,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        if num_mel_bins < 1:
            raise ValueError(f"num_mel_bins must be >= 1, got {num_mel_bins}")
        self.timeout = timeout
        self.window_size = int(window_size)
        self.overlap = float(overlap)
        self.num_mel_bins = int(num_mel_bins)
        self.gain = float(gain)
        self.bias = float(bias)
        self.time_constant = float(time_constant)
        self.power_to_db = bool(power_to_db)
        self.fmin = fmin
        self.fmax = fmax
        self.sigma = sigma

    @classmethod
    def from_config(cls, config, timeout: float | None = None) -> "PcenExtractor":
        """Build from :class:`~stm.config.Config` / ``config.yaml`` PCEN keys."""
        return cls(
            timeout=timeout,
            window_size=config.window_size,
            overlap=config.overlap,
            num_mel_bins=config.num_mel_bins,
            gain=config.pcen_gain,
            bias=config.pcen_bias,
            time_constant=config.pcen_time_constant,
            power_to_db=config.pcen_power_to_db,
            fmin=config.pcen_fmin,
            fmax=config.pcen_fmax,
            sigma=config.sigma,
        )

    @property
    def hop_length(self) -> int:
        return int(self.window_size * (1 - self.overlap))

    @property
    def feature_width(self) -> int:
        return self.num_mel_bins

    def _provenance(self, source: Path, **extra) -> dict:
        provenance = {
            "extractor": "PcenExtractor",
            "source": str(source),
            "window_size": self.window_size,
            "overlap": self.overlap,
            "num_mel_bins": self.num_mel_bins,
            "gain": self.gain,
            "bias": self.bias,
            "time_constant": self.time_constant,
            "power_to_db": self.power_to_db,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "sigma": self.sigma,
        }
        provenance.update(extra)
        return provenance

    def hop_seconds(self, sample_rate: float) -> float:
        return self.hop_length / float(sample_rate)

    def extract(self, source: Path, grid: TimeGrid | None = None) -> FeatureBlock:
        """PCEN frames for *source* (WAV or directory of WAVs).

        *grid* is accepted for API compatibility with :class:`Perch2Extractor`
        but does not pool frames; resolution is always the STFT hop.
        """
        if self.timeout is None:
            return self._extract(source)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._extract, source)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeout as exc:
                raise TimeoutError(
                    f"PcenExtractor.extract exceeded {self.timeout}s"
                ) from exc

    def _extract(self, source: Path) -> FeatureBlock:
        source = Path(source)
        wavs = _wav_paths(source)
        blocks: list[FeatureBlock] = []
        offset = 0.0
        for wav in wavs:
            block = self._extract_file(wav)
            if offset and len(block.grid):
                edges = np.asarray(block.grid.edges) + offset
                block = FeatureBlock(
                    values=block.values,
                    grid=TimeGrid(edges),
                    mask=block.mask,
                    provenance=block.provenance,
                )
            blocks.append(block)
            if len(block.grid):
                offset = float(block.grid.ends[-1])
        fused = _concat_file_blocks(
            blocks,
            provenance=self._provenance(
                source,
                sources=[block.provenance["source"] for block in blocks],
            ),
        )
        return fused

    def _extract_file(self, source: Path) -> FeatureBlock:
        signal, sr = sf.read(source.as_posix(), always_2d=False)
        if getattr(signal, "ndim", 1) > 1:
            signal = signal[:, 0]
        signal = np.asarray(signal, dtype=np.float64)
        nyquist = sr / 2.0
        fmin = 0.0 if self.fmin is None else float(self.fmin)
        fmax = nyquist if self.fmax is None else min(float(self.fmax), nyquist)
        pcen = compute_stft_pcen(
            signal,
            self.window_size,
            self.overlap,
            sr,
            fmin,
            fmax,
            num_mel_bins=self.num_mel_bins,
            gain=self.gain,
            bias=self.bias,
            tc=self.time_constant,
            power_to_db=self.power_to_db,
        )
        if self.sigma is not None:
            pcen = gaussian_filter(pcen, sigma=self.sigma)
        n_frames = int(pcen.shape[1])
        if n_frames == 0:
            grid = TimeGrid(np.zeros((0, 2), dtype=np.float64))
            values = np.zeros((0, self.num_mel_bins), dtype=np.float32)
            mask = np.zeros(0, dtype=bool)
        else:
            starts = librosa.frames_to_time(
                np.arange(n_frames),
                sr=sr,
                hop_length=self.hop_length,
            )
            win_s = self.window_size / float(sr)
            grid = TimeGrid(np.column_stack([starts, starts + win_s]))
            values = np.ascontiguousarray(pcen.T, dtype=np.float32)
            mask = np.ones(n_frames, dtype=bool)
        return FeatureBlock(
            values=values,
            grid=grid,
            mask=mask,
            provenance=self._provenance(source, sample_rate=int(sr)),
        )

    def run(
        self,
        source: Path,
        grids: Sequence[TimeGrid],
        target: TimeGrid | None = None,
    ) -> FeatureBlock | list[FeatureBlock]:
        """Extract on each grid; :func:`combine` onto *target* when given."""
        if not grids:
            raise ValueError("run() requires at least one TimeGrid")
        source = Path(source)
        blocks = [self.extract(source, grid) for grid in grids]
        if target is not None:
            return combine(blocks, target)
        if len(blocks) == 1:
            return blocks[0]
        return blocks

