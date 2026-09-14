# stm, Apache-2.0 license
# Filename: test_features.py
# Description: Tests for TimeGrid, FeatureBlock, Perch2Extractor, and PcenExtractor
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from perchtopic.features import (
    FeatureBlock,
    PcenExtractor,
    Perch2Extractor,
    TimeGrid,
    combine,
    compute_stft_pcen,
    pool,
)


def test_timegrid_regular_edges() -> None:
    grid = TimeGrid.regular(start=1.0, hop=0.5, window=2.0, n=3)
    assert len(grid) == 3
    np.testing.assert_allclose(
        grid.edges,
        np.array([[1.0, 3.0], [1.5, 3.5], [2.0, 4.0]]),
    )
    np.testing.assert_allclose(grid.durations, 2.0)


def test_timegrid_regular_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="hop"):
        TimeGrid.regular(0.0, hop=0.0, window=1.0, n=1)
    with pytest.raises(ValueError, match="window"):
        TimeGrid.regular(0.0, hop=1.0, window=0.0, n=1)
    with pytest.raises(ValueError, match="n must"):
        TimeGrid.regular(0.0, hop=1.0, window=1.0, n=-1)


def test_feature_block_shape_checks() -> None:
    grid = TimeGrid.regular(0.0, hop=1.0, window=5.0, n=2)
    with pytest.raises(ValueError, match="does not match grid"):
        FeatureBlock(values=np.zeros((3, 4)), grid=grid)


def test_perch_run_requires_grids(tmp_path: Path) -> None:
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    extractor = Perch2Extractor(model=model, timeout=120.0)
    assert extractor.timeout == 120.0
    with pytest.raises(ValueError, match="at least one TimeGrid"):
        extractor.run(Path("x.wav"), grids=[])


def test_extract_requires_model_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.onnx"
    with pytest.raises(ValueError, match="does not exist"):
        Perch2Extractor(model=missing)
    directory = tmp_path / "not_a_model"
    directory.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        Perch2Extractor(model=directory)


def test_extract_uses_embed_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sr = 32000
    audio = np.linspace(-0.2, 0.2, sr, dtype=np.float32)
    wav = tmp_path / "a.wav"
    sf.write(wav.as_posix(), audio, sr)
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")

    captured: dict = {}

    def fake_session(model_path):
        captured["model_path"] = Path(model_path)
        return object()

    def fake_embed(windows, session, batch_size=32, **kwargs):
        captured["windows"] = np.asarray(windows)
        captured["batch_size"] = batch_size
        return np.arange(len(windows) * 1536, dtype=np.float32).reshape(len(windows), 1536)

    monkeypatch.setattr("stm.features.load_onnx_session", fake_session)
    monkeypatch.setattr("stm.features.embed_windows", fake_embed)

    extractor = Perch2Extractor(model=model, window_fill="pad")
    grid = TimeGrid.regular(start=0.0, hop=0.5, window=1.0, n=3)
    # last interval starts at 1.0s; file is 1.0s so [1.0, 2.0) is empty
    block = extractor.extract(wav, grid)

    assert captured["windows"].shape == (2, 160000)
    assert captured["batch_size"] == 32
    assert block.values.shape == (3, 1536)
    np.testing.assert_array_equal(block.mask, [True, True, False])
    np.testing.assert_array_equal(block.values[2], 0)
    assert block.provenance["extractor"] == "Perch2Extractor"
    assert block.provenance["source"] == str(wav)
    assert (tmp_path / "perch_raw" / "hop_0.5_audio_1_fill-pad" / "a.values.npy").is_file()


def test_extract_directory_rglob_wavs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sr = 32000
    nested = tmp_path / "units" / "a"
    nested.mkdir(parents=True)
    sf.write((tmp_path / "units" / "b.wav").as_posix(), np.ones(sr, dtype=np.float32), sr)
    sf.write((nested / "a.wav").as_posix(), np.full(sr, 0.5, dtype=np.float32), sr)
    (tmp_path / "units" / "skip.txt").write_text("not audio")

    def fake_session(model_path):
        return object()

    def fake_embed(windows, session, batch_size=32, **kwargs):
        return np.ones((len(windows), 1536), dtype=np.float32)

    monkeypatch.setattr("stm.features.load_onnx_session", fake_session)
    monkeypatch.setattr("stm.features.embed_windows", fake_embed)

    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    extractor = Perch2Extractor(model=model)
    grid = TimeGrid.regular(start=0.0, hop=1.0, window=1.0, n=1)
    block = extractor.extract(tmp_path / "units", grid)

    assert block.values.shape == (2, 1536)
    np.testing.assert_array_equal(block.mask, [True, True])
    assert block.provenance["source"] == str(tmp_path / "units")
    assert [Path(p).name for p in block.provenance["sources"]] == ["a.wav", "b.wav"]


def test_extract_empty_directory(tmp_path: Path) -> None:
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    extractor = Perch2Extractor(model=model)
    grid = TimeGrid.regular(0.0, hop=1.0, window=1.0, n=1)
    with pytest.raises(FileNotFoundError, match="No wav clips"):
        extractor.extract(tmp_path, grid)


def test_extract_writes_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 32000
    wav = tmp_path / "a.wav"
    sf.write(wav.as_posix(), np.linspace(-0.2, 0.2, sr, dtype=np.float32), sr)
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    output = tmp_path / "out"
    calls = {"n": 0}

    def fake_embed(windows, session, batch_size=32, **kwargs):
        calls["n"] += 1
        return np.arange(len(windows) * 1536, dtype=np.float32).reshape(len(windows), 1536)

    monkeypatch.setattr("stm.features.load_onnx_session", lambda model_path: object())
    monkeypatch.setattr("stm.features.embed_windows", fake_embed)

    extractor = Perch2Extractor(model=model, window_fill="pad", output_path=output)
    grid = TimeGrid.regular(start=0.0, hop=0.5, window=1.0, n=3)
    first = extractor.extract(wav, grid)
    second = extractor.extract(wav, grid)

    assert calls["n"] == 1
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.mask, second.mask)
    cache_dir = output / "perch_raw" / "hop_0.5_audio_1_fill-pad"
    assert (cache_dir / "a.values.npy").is_file()
    assert (cache_dir / "a.metadata.json").is_file()
    assert second.provenance["cache_dir"] == str(cache_dir)


def test_extract_cache_miss_on_window_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 32000
    wav = tmp_path / "a.wav"
    sf.write(wav.as_posix(), np.ones(sr, dtype=np.float32), sr)
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    output = tmp_path / "out"
    calls = {"n": 0}

    def fake_embed(windows, session, batch_size=32, **kwargs):
        calls["n"] += 1
        return np.ones((len(windows), 1536), dtype=np.float32) * calls["n"]

    monkeypatch.setattr("stm.features.load_onnx_session", lambda model_path: object())
    monkeypatch.setattr("stm.features.embed_windows", fake_embed)

    grid = TimeGrid.regular(start=0.0, hop=1.0, window=1.0, n=1)
    pad = Perch2Extractor(model=model, window_fill="pad", output_path=output)
    tile = Perch2Extractor(model=model, window_fill="tile", output_path=output)
    pad.extract(wav, grid)
    tile.extract(wav, grid)
    assert calls["n"] == 2
    assert (output / "perch_raw" / "hop_1_audio_1_fill-pad").is_dir()
    assert (output / "perch_raw" / "hop_1_audio_1_fill-tile").is_dir()


def test_extract_resumes_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 32000
    wav = tmp_path / "a.wav"
    sf.write(wav.as_posix(), np.ones(2 * sr, dtype=np.float32), sr)
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    output = tmp_path / "out"
    calls = {"n": 0}

    def fake_embed(windows, session, batch_size=32, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("stop after first batch")
        return np.full((len(windows), 1536), calls["n"], dtype=np.float32)

    monkeypatch.setattr("stm.features.load_onnx_session", lambda model_path: object())
    monkeypatch.setattr("stm.features.embed_windows", fake_embed)

    extractor = Perch2Extractor(
        model=model, window_fill="pad", output_path=output, batch_size=1
    )
    grid = TimeGrid.regular(start=0.0, hop=1.0, window=1.0, n=2)
    with pytest.raises(RuntimeError, match="stop after first batch"):
        extractor.extract(wav, grid)

    block = extractor.extract(wav, grid)
    assert calls["n"] == 3
    np.testing.assert_array_equal(block.values[0], 1)
    np.testing.assert_array_equal(block.values[1], 3)
    assert block.mask.all()


def test_extract_passes_progress_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 32000
    wav = tmp_path / "a.wav"
    sf.write(wav.as_posix(), np.ones(3 * sr, dtype=np.float32), sr)
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    seen: list[dict] = []

    def fake_embed(windows, session, batch_size=32, **kwargs):
        seen.append({"n": len(windows), **kwargs})
        return np.ones((len(windows), 1536), dtype=np.float32)

    monkeypatch.setattr("stm.features.load_onnx_session", lambda model_path: object())
    monkeypatch.setattr("stm.features.embed_windows", fake_embed)

    extractor = Perch2Extractor(
        model=model, window_fill="pad", output_path=tmp_path / "out", batch_size=1
    )
    grid = TimeGrid.regular(start=0.0, hop=1.0, window=1.0, n=3)
    extractor.extract(wav, grid)

    assert [row["progress_offset"] for row in seen] == [0, 1, 2]
    assert [row["progress_total"] for row in seen] == [3, 3, 3]


def test_pool_mean_std_and_combine() -> None:
    src_grid = TimeGrid.regular(start=0.0, hop=1.0, window=1.0, n=2)
    target = TimeGrid.regular(start=0.0, hop=2.0, window=2.0, n=1)
    values = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    block = FeatureBlock(values=values, grid=src_grid)

    pooled = pool(block, target)
    assert pooled.values.shape == (1, 4)
    np.testing.assert_allclose(pooled.values[0, :2], [1.0, 3.0])
    np.testing.assert_allclose(pooled.values[0, 2:], [1.0, 1.0])
    np.testing.assert_array_equal(pooled.mask, [True])

    aligned = pool(block, src_grid)
    np.testing.assert_allclose(aligned.values[:, :2], values)
    np.testing.assert_allclose(aligned.values[:, 2:], 0.0)

    combined = combine([block, block], target)
    assert combined.values.shape == (1, 8)
    np.testing.assert_allclose(combined.values[0, :4], pooled.values[0])
    np.testing.assert_allclose(combined.values[0, 4:], pooled.values[0])

    with pytest.raises(ValueError, match="at least one"):
        combine([], target)


def test_pool_no_overlap_is_masked() -> None:
    src = FeatureBlock(
        values=np.ones((1, 2), dtype=np.float32),
        grid=TimeGrid.regular(0.0, hop=1.0, window=1.0, n=1),
    )
    target = TimeGrid.regular(start=10.0, hop=1.0, window=1.0, n=1)
    pooled = pool(src, target)
    np.testing.assert_array_equal(pooled.mask, [False])
    np.testing.assert_array_equal(pooled.values, 0)


def test_compute_stft_pcen_shape() -> None:
    sr = 32000
    t = np.arange(sr, dtype=np.float64) / sr
    signal = np.sin(2 * np.pi * 1000 * t)
    pcen = compute_stft_pcen(
        signal, window_size=512, overlap=0.5, fs=sr, fmin=0.0, fmax=sr / 2, num_mel_bins=32
    )
    assert pcen.ndim == 2
    assert pcen.shape[0] == 32
    assert pcen.shape[1] > 1
    assert np.isfinite(pcen).all()


def test_compute_stft_pcen_constant_signal() -> None:
    pcen = compute_stft_pcen(
        np.zeros(32000, dtype=np.float64),
        window_size=512,
        overlap=0.5,
        fs=32000,
        fmin=0.0,
        fmax=16000.0,
        num_mel_bins=16,
    )
    assert pcen.shape[0] == 16
    assert np.isfinite(pcen).all()


def test_extractor_feature_width(tmp_path: Path) -> None:
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    assert Perch2Extractor(model=model).feature_width() == 1536
    assert PcenExtractor().feature_width() == 32
    assert PcenExtractor(num_mel_bins=16).feature_width() == 16
    assert PcenExtractor(cluster="kmeans").feature_width() == 1


def test_pcen_run_requires_grids() -> None:
    extractor = PcenExtractor(timeout=30.0)
    assert extractor.timeout == 30.0
    assert extractor.embedding_dim == 32
    assert extractor.feature_width() == 32
    with pytest.raises(ValueError, match="at least one TimeGrid"):
        extractor.run(Path("x.wav"), grids=[])


def test_pcen_extract_native_stft_frames(tmp_path: Path) -> None:
    sr = 32000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    wav = tmp_path / "a.wav"
    sf.write(wav.as_posix(), audio, sr)

    extractor = PcenExtractor(num_mel_bins=16, window_size=512, overlap=0.5, sigma=None)
    block = extractor.extract(wav)

    assert block.values.ndim == 2
    assert block.values.shape[1] == 16
    assert block.values.shape[0] > 10
    assert len(block.grid) == block.values.shape[0]
    np.testing.assert_array_equal(block.mask, np.ones(len(block.grid), dtype=bool))
    assert np.isfinite(block.values).all()
    assert not np.allclose(block.values[0], block.values[len(block.values) // 2])
    assert block.provenance["extractor"] == "PcenExtractor"
    assert block.provenance["source"] == str(wav)
    assert block.provenance["num_mel_bins"] == 16
    np.testing.assert_allclose(block.grid.durations, 512 / sr)


def test_pcen_extract_directory_rglob_wavs(tmp_path: Path) -> None:
    sr = 32000
    nested = tmp_path / "units" / "a"
    nested.mkdir(parents=True)
    tone = (0.1 * np.sin(2 * np.pi * 800 * np.arange(sr) / sr)).astype(np.float32)
    sf.write((tmp_path / "units" / "b.wav").as_posix(), tone, sr)
    sf.write((nested / "a.wav").as_posix(), tone * 0.5, sr)
    (tmp_path / "units" / "skip.txt").write_text("not audio")

    extractor = PcenExtractor(num_mel_bins=8, sigma=None)
    block = extractor.extract(tmp_path / "units")

    one = extractor.extract(nested / "a.wav")
    assert block.values.shape[1] == 8
    assert block.values.shape[0] == 2 * one.values.shape[0]
    assert block.mask.all()
    assert block.provenance["source"] == str(tmp_path / "units")
    assert [Path(p).name for p in block.provenance["sources"]] == ["a.wav", "b.wav"]
    assert float(block.grid.starts[-1]) > float(one.grid.starts[-1])


def test_pcen_extract_empty_directory(tmp_path: Path) -> None:
    extractor = PcenExtractor()
    grid = TimeGrid.regular(0.0, hop=1.0, window=1.0, n=1)
    with pytest.raises(FileNotFoundError, match="No wav clips"):
        extractor.extract(tmp_path, grid)


def test_kmeans_word_ids_separates_blobs() -> None:
    values = np.vstack(
        [
            np.zeros((8, 4), dtype=np.float32),
            np.ones((8, 4), dtype=np.float32) * 10,
        ]
    )
    labels, vocab = kmeans_word_ids(values, n_clusters=2, random_state=0)
    assert labels.shape == (16,)
    assert vocab == 2
    assert set(labels[:8]) != set(labels[8:])
    assert len(set(labels[:8])) == 1
    assert len(set(labels[8:])) == 1


def test_kmeans_cluster_grid_logspaced() -> None:
    ks = kmeans_cluster_grid(10_000)
    assert ks[0] == 2
    assert ks[-1] == 10_000
    assert len(ks) == 10
    assert kmeans_cluster_grid(40, k_grid=[2, 3, 8, 99]).tolist() == [2, 3, 8]


def test_knee_n_clusters_picks_elbow() -> None:
    ks = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    inertias = np.array([100.0, 50.0, 20.0, 8.0, 7.2, 6.8, 6.5, 6.3])
    assert knee_n_clusters(ks, inertias) == 16


def test_kmeans_word_ids_knee_finds_three_blobs() -> None:
    rng = np.random.default_rng(0)
    values = np.vstack(
        [
            np.zeros((30, 6), dtype=np.float32),
            np.full((30, 6), 10.0, dtype=np.float32),
            np.full((30, 6), -10.0, dtype=np.float32),
        ]
    )
    values = values + 0.05 * rng.normal(size=values.shape).astype(np.float32)
    labels, vocab = kmeans_word_ids(
        values, n_clusters=None, random_state=0, k_grid=[2, 3, 4, 6, 8, 12]
    )
    assert vocab == 3
    assert labels.shape == (90,)
    groups = [set(labels[0:30]), set(labels[30:60]), set(labels[60:90])]
    assert all(len(group) == 1 for group in groups)
    assert len(set.union(*groups)) == 3


def test_pcen_cluster_option_validation() -> None:
    with pytest.raises(ValueError, match="None or 'kmeans'"):
        PcenExtractor(cluster="hdbscan")
    with pytest.raises(ValueError, match="n_clusters must be"):
        PcenExtractor(cluster="kmeans", n_clusters=1)
    extractor = PcenExtractor(cluster="kmeans")
    assert extractor.n_clusters is None
    extractor = PcenExtractor(cluster="kmeans", n_clusters=4)
    assert extractor.embedding_dim == 1
    assert extractor.feature_width() == 1
    assert extractor.cluster == "kmeans"
    assert extractor.n_clusters == 4


def test_pcen_extract_kmeans_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    grid = TimeGrid.regular(0.0, hop=0.008, window=0.016, n=40)
    raw = FeatureBlock(
        values=np.vstack(
            [
                np.zeros((20, 8), dtype=np.float32),
                np.full((20, 8), 10.0, dtype=np.float32),
            ]
        ),
        grid=grid,
        provenance={"extractor": "PcenExtractor", "source": str(wav)},
    )
    monkeypatch.setattr(PcenExtractor, "_extract_file", lambda self, source: raw)

    extractor = PcenExtractor(
        num_mel_bins=8,
        sigma=None,
        cluster="kmeans",
        n_clusters=2,
    )
    block = extractor.extract(wav)

    assert block.values.shape == (40, 1)
    ids = np.rint(block.values[:, 0]).astype(np.int64)
    assert set(ids[:20]) != set(ids[20:])
    assert len(set(ids)) == 2
    assert block.provenance["cluster"] == "kmeans"
    assert block.provenance["n_clusters"] == 2


def test_pcen_extract_kmeans_knee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    grid = TimeGrid.regular(0.0, hop=0.008, window=0.016, n=90)
    raw = FeatureBlock(
        values=(
            np.vstack(
                [
                    np.zeros((30, 8), dtype=np.float32),
                    np.full((30, 8), 10.0, dtype=np.float32),
                    np.full((30, 8), -10.0, dtype=np.float32),
                ]
            )
            + 0.05 * np.random.default_rng(0).normal(size=(90, 8)).astype(np.float32)
        ),
        grid=grid,
        provenance={"extractor": "PcenExtractor", "source": str(wav)},
    )
    monkeypatch.setattr(PcenExtractor, "_extract_file", lambda self, source: raw)
    monkeypatch.setattr(
        "stm.features.kmeans_cluster_grid",
        lambda n_samples, k_grid=None: np.array([2, 3, 4, 6, 8, 12], dtype=int),
    )

    extractor = PcenExtractor(num_mel_bins=8, sigma=None, cluster="kmeans")
    block = extractor.extract(wav)

    assert extractor.n_clusters is None
    assert block.values.shape == (90, 1)
    ids = np.rint(block.values[:, 0]).astype(np.int64)
    assert len(set(ids)) == 3
    assert block.provenance["cluster"] == "kmeans"
    assert block.provenance["n_clusters"] == 3
    assert block.provenance["n_clusters_method"] == "knee"

