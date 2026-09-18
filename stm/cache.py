# stm, Apache-2.0 license
# Filename: cache.py
# Description: Cache keys for feature-extraction caches.
"""Cache keys for feature-extraction caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stm.config import Config


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* as JSON, replacing *path* only after a full flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Write *array* as ``.npy``, replacing *path* only after a full flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def perch_cache_stem(wav: Path, source: Path) -> str:
    """Stable filename for one WAV under an extract *source* (file or directory)."""
    wav = Path(wav).resolve()
    root = Path(source).resolve()
    if root.is_file():
        root = root.parent
    try:
        relative = wav.relative_to(root)
    except ValueError:
        return wav.stem
    return "__".join(relative.with_suffix("").parts)


def default_perch_cache_root(source: Path) -> Path:
    """Dataset directory so ``perch_raw/`` sits next to the WAVs when no output_path is set."""
    source = Path(source).resolve()
    return source if source.is_dir() else source.parent


@dataclass(frozen=True)
class UnitCacheKey:
    """Fingerprint for the raw unit-embedding cache tier.

    Fields mirror :class:`~stm.config.Config` so the JSON sidecar
    (``as_dict``) and cache directory name (``dir_label``) stay in sync.
    """

    wav_path: Path
    output_path: Path
    perch_hop_seconds: float = 1.0
    perch_audio_seconds: float = 1.0
    perch_window_fill: str = "fill"

    def as_dict(self) -> dict[str, Any]:
        """Metadata written to ``unit_metadata.json``."""
        return {
            "wav_path": str(self.wav_path),
            "output_path": str(self.output_path),
            "perch_hop_seconds": self.perch_hop_seconds,
            "perch_audio_seconds": self.perch_audio_seconds,
            "perch_window_fill": self.perch_window_fill,
        }

    def dir_label(self) -> str:
        """Folder name under ``units_raw/`` for this cache key."""
        return (
            f"hop_{self.perch_hop_seconds:g}"
            f"_audio_{self.perch_audio_seconds:g}"
            f"_fill-{self.perch_window_fill}"
        )

    def cache_dir(self, base: Path | None = None) -> Path:
        """Directory for raw unit embeddings keyed by crop settings."""
        root = self.output_path if base is None else Path(base)
        return root / "units_raw" / self.dir_label()

    def matches(self, stored: dict[str, Any]) -> bool:
        """True if *stored* sidecar metadata matches this key."""
        for key, value in self.as_dict().items():
            if stored.get(key) != value:
                return False
        return True

    @classmethod
    def from_config(cls, config: Config) -> UnitCacheKey:
        """Build a key from a :class:`Config` instance."""
        return cls(
            wav_path=config.wav_path,
            output_path=config.output_path,
            perch_hop_seconds=config.perch_hop_seconds,
            perch_audio_seconds=config.perch_audio_seconds,
            perch_window_fill=config.perch_window_fill,
        )


@dataclass(frozen=True)
class PerchCacheKey:
    """Fingerprint for :class:`~stm.features.Perch2Extractor` window embeddings.

    Directory name follows the unit-cache pattern (hop / audio / fill). The
    JSON sidecar also records grid length, model path, and WAV identity so a
    stale file is never reused.
    """

    wav_path: Path
    output_path: Path
    window_fill: str
    model: str
    grid_start: float
    grid_hop: float | None
    grid_window: float | None
    grid_n: int
    wav_mtime_ns: int
    wav_size: int
    edges_sha1: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Metadata written next to ``*.values.npy``."""
        return {
            "wav_path": str(self.wav_path),
            "output_path": str(self.output_path),
            "window_fill": self.window_fill,
            "model": self.model,
            "grid_start": self.grid_start,
            "grid_hop": self.grid_hop,
            "grid_window": self.grid_window,
            "grid_n": self.grid_n,
            "wav_mtime_ns": str(self.wav_mtime_ns),
            "wav_size": self.wav_size,
            "edges_sha1": self.edges_sha1,
        }

    def dir_label(self) -> str:
        """Folder name under ``perch_raw/`` for this cache key."""
        if self.edges_sha1:
            return f"edges_{self.edges_sha1}_fill-{self.window_fill}"
        hop = 0.0 if self.grid_hop is None else self.grid_hop
        window = 0.0 if self.grid_window is None else self.grid_window
        return f"hop_{hop:g}_audio_{window:g}_fill-{self.window_fill}"

    def cache_dir(self, base: Path | None = None) -> Path:
        """Directory for Perch2 window embeddings keyed by grid settings."""
        root = self.output_path if base is None else Path(base)
        return root / "perch_raw" / self.dir_label()

    def matches(self, stored: dict[str, Any]) -> bool:
        """True if *stored* sidecar metadata matches this key."""
        for key, value in self.as_dict().items():
            if stored.get(key) != value:
                return False
        return True
