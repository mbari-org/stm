# stm, Apache-2.0 license
# Filename: test_plotter.py
# Description: Tests for Plotter model-directory checks and chunk plots
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import soundfile as sf

from perchtopic.features import FeatureBlock, TimeGrid
from perchtopic.topicmodel.plotter import Plotter


def _write_theta(model_dir: Path, n_docs: int = 10, n_topics: int = 3) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    theta = np.full((n_docs, n_topics), 1.0 / n_topics)
    theta[:, 0] = 0.8
    theta[:, 1:] = 0.1
    path = model_dir / "theta.csv"
    np.savetxt(path, theta, delimiter=",")
    return path


def test_plotter_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Plotter(tmp_path / "missing")


def test_plotter_path_is_file(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        Plotter(target)


def test_plotter_missing_theta(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing theta.csv"):
        Plotter(model_dir)


def test_plotter_loads_theta(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_theta(model_dir, n_docs=4, n_topics=2)
    plotter = Plotter(model_dir)
    assert plotter.theta.shape == (4, 2)
    assert plotter.theta_path == model_dir / "theta.csv"


def test_plot_writes_first_two_chunks(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_theta(model_dir, n_docs=10, n_topics=3)
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(32000 * 10, dtype=np.float32), 32000)
    plotter = Plotter(model_dir)
    saved = plotter.plot(
        wav,
        hop=1.0,
        window=1.0,
        chunk_size=4.0,
        overlap_fraction=0.25,
        n_chunks=2,
        show=False,
    )
    assert len(saved) == 2
    assert all(path.is_file() for path in saved)
    assert saved[0].name == "clip_000000_000004_topics_.png"
    assert saved[1].name == "clip_000003_000007_topics_.png"


def test_plot_accepts_feature_block_grid(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_theta(model_dir, n_docs=6, n_topics=2)
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(32000 * 8, dtype=np.float32), 32000)
    grid = TimeGrid.regular(start=0.0, hop=1.0, window=1.0, n=6)
    block = FeatureBlock(values=np.zeros((6, 4), dtype=np.float32), grid=grid)
    plotter = Plotter(model_dir)
    saved = plotter.plot(
        wav,
        grid=block,
        chunk_size=3.0,
        overlap_fraction=0.25,
        n_chunks=1,
        show=False,
    )
    assert len(saved) == 1
    assert saved[0].is_file()
