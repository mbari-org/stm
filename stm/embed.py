# stm, Apache-2.0 license
# Filename: embed.py
# Description: ONNX embeddings for exported unit clips
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from stm.cache import UnitCacheKey

PERCH_ONNX_URL = "https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx"
PERCH_INPUT_SAMPLES = 160000  # Perch v2: 5 s at 32 kHz


def _ensure_onnx_model(model_path: Path) -> Path:
    """Download perch_v2.onnx from Hugging Face when *model_path* is missing."""
    if model_path.exists():
        return model_path

    print(f"{model_path} not found; downloading {PERCH_ONNX_URL}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".part")
    request = urllib.request.Request(PERCH_ONNX_URL, headers={"User-Agent": "stm"})
    try:
        with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            copied = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                copied += len(chunk)
                if total:
                    print(f"  Downloaded {copied / 1e6:.1f} / {total / 1e6:.1f} MB", end="\r")
                else:
                    print(f"  Downloaded {copied / 1e6:.1f} MB", end="\r")
        tmp_path.replace(model_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    print(f"\nSaved {model_path}")
    return model_path


def _fill_to_length(segment: np.ndarray, n: int, fill: str) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32)
    if len(segment) >= n:
        return segment[:n]
    if fill == "tile":
        return np.resize(segment, n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: len(segment)] = segment
    return out


def load_onnx_session(model_path: Path) -> ort.InferenceSession:
    """Load (or download) perch_v2.onnx and return an inference session."""
    model_path = _ensure_onnx_model(Path(model_path))
    print(f"Loading ONNX model {model_path}")
    return ort.InferenceSession(model_path.as_posix())


def _session_io(session: ort.InferenceSession) -> tuple[str, str]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for output in session.get_outputs():
        if output.name == "embedding":
            output_name = output.name
            break
    return input_name, output_name


def embed_windows(
    windows: np.ndarray,
    session: ort.InferenceSession,
    batch_size: int = 32,
    *,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> np.ndarray:
    """Embed a ``(N, PERCH_INPUT_SAMPLES)`` float32 batch. Returns ``(N, D)``.

    *progress_offset* / *progress_total* are the clip indices in the full
    extract, so a caller that feeds one ONNX batch at a time still reports
    ``1-256 of 768`` instead of ``1-256 of 256``.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 2 or windows.shape[1] != PERCH_INPUT_SAMPLES:
        raise ValueError(
            f"windows must have shape (N, {PERCH_INPUT_SAMPLES}), got {windows.shape}"
        )
    input_name, output_name = _session_io(session)
    parts = []
    n = len(windows)
    total = n if progress_total is None else int(progress_total)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        print(
            f"Embedding clips {progress_offset + lo + 1}-{progress_offset + hi} of {total}"
        )
        parts.append(
            np.asarray(
                session.run([output_name], {input_name: windows[lo:hi]})[0],
                dtype=np.float32,
            )
        )
    return np.concatenate(parts, axis=0)


def total_audio_seconds(dataset_path: Path | str) -> float:
    """Return the total duration in seconds of WAV files under *dataset_path*.

    *dataset_path* may be a dataset directory or a single WAV file.
    """
    path = Path(dataset_path)
    if path.is_file():
        if path.suffix.lower() != ".wav":
            return 0.0
        return float(sf.info(path.as_posix()).duration)
    if not path.is_dir():
        return 0.0
    return sum(
        float(sf.info(clip.as_posix()).duration)
        for clip in sorted(path.rglob("*.wav"))
    )


def _write_unit_metadata(meta_path: Path, key: UnitCacheKey) -> dict:
    metadata = key.as_dict()
    metadata["total_audio_seconds"] = total_audio_seconds(key.wav_path)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class Embedding:
    """ONNX embeddings for unit clips, keyed by :class:`UnitCacheKey`."""

    def __init__(self, key: UnitCacheKey, model_path: Path, batch_size: int = 32):
        self.key = key
        self.model_path = Path(model_path)
        self.batch_size = batch_size

    def run(self) -> np.ndarray:
        cache_dir = self.key.cache_dir()
        emb_path = cache_dir / "embeddings.npy"
        meta_path = cache_dir / "unit_metadata.json"

        if emb_path.exists() and meta_path.exists():
            stored = json.loads(meta_path.read_text())
            if self.key.matches(stored):
                embeddings = np.load(emb_path)
                if "total_audio_seconds" not in stored:
                    _write_unit_metadata(meta_path, self.key)
                print(f"Using cached embeddings {embeddings.shape} in {cache_dir}")
                return embeddings

        clips = sorted((self.key.output_path / "units").rglob("*.wav"))
        if not clips:
            raise FileNotFoundError(f"No wav clips under {self.key.output_path / 'units'}")

        session = load_onnx_session(self.model_path)

        print(
            f"Loading {len(clips)} clips; "
            f"keep {self.key.perch_audio_seconds:g}s then "
            f"{self.key.perch_window_fill} to {PERCH_INPUT_SAMPLES} samples"
        )
        windows = []
        for i, clip in enumerate(clips, 1):
            print(f"  [{i}/{len(clips)}] {clip.name}")
            audio, sr = sf.read(clip.as_posix(), always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio[:, 0]
            keep = int(round(self.key.perch_audio_seconds * sr))
            windows.append(
                _fill_to_length(np.asarray(audio[:keep], dtype=np.float32), PERCH_INPUT_SAMPLES, self.key.perch_window_fill)
            )

        batch = np.stack(windows).astype(np.float32)
        print(f"ONNX input batch shape {batch.shape}")
        embeddings = embed_windows(batch, session, batch_size=self.batch_size)

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        _write_unit_metadata(meta_path, self.key)
        print(f"Wrote {embeddings.shape} embeddings to {cache_dir}")
        return embeddings
