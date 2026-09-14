# stm, Apache-2.0 license
# Filename: test_embed.py
# Description: Tests for unit embedding cache metadata
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from perchtopic.cache import UnitCacheKey
from perchtopic.embed import PERCH_INPUT_SAMPLES, Embedding, embed_windows, total_audio_seconds


def _write_wav(path: Path, n_samples: int, sr: int = 32000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path.as_posix(), np.zeros(n_samples, dtype=np.float32), sr)


def test_total_audio_seconds_sums_directory_wavs(tmp_path: Path) -> None:
    sr = 32000
    _write_wav(tmp_path / "a.wav", sr)
    _write_wav(tmp_path / "nested" / "b.wav", 2 * sr)
    (tmp_path / "skip.txt").write_text("not audio")
    np.testing.assert_allclose(total_audio_seconds(tmp_path), 3.0)


def test_total_audio_seconds_single_file(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _write_wav(wav, 16000, sr=32000)
    np.testing.assert_allclose(total_audio_seconds(wav), 0.5)


def test_total_audio_seconds_accepts_str_path(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav", 32000)
    np.testing.assert_allclose(total_audio_seconds(str(tmp_path)), 1.0)


def test_total_audio_seconds_missing_path(tmp_path: Path) -> None:
    assert total_audio_seconds(tmp_path / "missing") == 0.0


def test_embed_windows_progress_uses_file_total(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("stm.embed._session_io", lambda session: ("in", "embedding"))

    class Session:
        def run(self, names, feed):
            n = len(next(iter(feed.values())))
            return [np.zeros((n, 2), dtype=np.float32)]

    windows = np.zeros((5, PERCH_INPUT_SAMPLES), dtype=np.float32)
    embed_windows(
        windows, Session(), batch_size=2, progress_offset=256, progress_total=768
    )
    out = capsys.readouterr().out
    assert "Embedding clips 257-258 of 768" in out
    assert "Embedding clips 259-260 of 768" in out
    assert "Embedding clips 261-261 of 768" in out
    assert "of 256" not in out


def test_run_writes_total_audio_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 32000
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_wav(dataset / "src.wav", 3 * sr, sr)
    _write_wav(output / "units" / "clip.wav", sr, sr)

    monkeypatch.setattr("stm.embed.load_onnx_session", lambda model_path: object())
    monkeypatch.setattr(
        "stm.embed.embed_windows",
        lambda windows, session, batch_size=32: np.ones((len(windows), 1536), dtype=np.float32),
    )

    key = UnitCacheKey(wav_path=dataset, output_path=output, perch_audio_seconds=1.0)
    embeddings = Embedding(key, tmp_path / "perch_v2.onnx").run()
    assert embeddings.shape == (1, 1536)

    meta = json.loads((key.cache_dir() / "unit_metadata.json").read_text())
    np.testing.assert_allclose(meta["total_audio_seconds"], 3.0)
    assert meta["wav_path"] == str(dataset)


def test_cache_hit_backfills_total_audio_seconds(tmp_path: Path) -> None:
    sr = 32000
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_wav(dataset / "src.wav", 2 * sr, sr)

    key = UnitCacheKey(wav_path=dataset, output_path=output)
    cache_dir = key.cache_dir()
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "embeddings.npy", np.ones((2, 1536), dtype=np.float32))
    (cache_dir / "unit_metadata.json").write_text(json.dumps(key.as_dict(), indent=2))

    embeddings = Embedding(key, tmp_path / "perch_v2.onnx").run()
    assert embeddings.shape == (2, 1536)

    meta = json.loads((cache_dir / "unit_metadata.json").read_text())
    np.testing.assert_allclose(meta["total_audio_seconds"], 2.0)
