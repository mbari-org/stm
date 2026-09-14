# stm, Apache-2.0 license
# Filename: test_topic_model.py
# Description: Tests for TopicModelRunner document writing and ROST command
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perchtopic.features import FeatureBlock, TimeGrid
from perchtopic.topicmodel.runner import TopicModelRunner, _singleton_noise


def test_write_documents_and_noise_singletons(tmp_path: Path) -> None:
    grid = TimeGrid.regular(start=0.0, hop=5.0, window=5.0, n=3)
    runner = TopicModelRunner(use_docker=True, words_per_doc=1)
    path, vocab = runner.write_documents(
        np.array([0, -1, 1]),
        grid,
        tmp_path / "all_docs.csv",
    )
    lines = path.read_text().strip().splitlines()
    assert lines == ["0,0", "5000,2", "10000,1"]
    assert vocab == 3


def test_write_documents_respects_mask(tmp_path: Path) -> None:
    grid = TimeGrid.regular(start=0.0, hop=1.0, window=1.0, n=3)
    runner = TopicModelRunner()
    path, vocab = runner.write_documents(
        np.array([7, 8]),
        grid,
        tmp_path / "docs.csv",
        mask=np.array([True, False, True]),
    )
    lines = path.read_text().strip().splitlines()
    assert lines == ["0,7", "2000,8"]
    assert vocab == 9


def test_singleton_noise_all_noise() -> None:
    np.testing.assert_array_equal(_singleton_noise(np.array([-1, -1])), [0, 1])


def test_run_builds_docker_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "docs" / "all_docs.csv"
    docs.parent.mkdir()
    docs.write_text("0,0\n5000,1\n")
    model = tmp_path / "model"
    captured: list[list[str]] = []

    def fake_execute(cmd, mount, log_path=None):
        captured.append(list(cmd))
        return ""

    runner = TopicModelRunner(timeout=120.0, num_topics=4, threads=2)
    monkeypatch.setattr(runner, "_execute", fake_execute)
    result = runner.run(docs, model, vocab_size=2)

    assert captured[0][0] == "topics.refine.t"
    assert "-K" in captured[0]
    assert captured[0][captured[0].index("-K") + 1] == "4"
    assert captured[0][captured[0].index("-V") + 1] == "2"
    assert str(docs.resolve()) in captured[0]
    assert result.vocab_size == 2
    assert result.num_topics == 4
    assert result.topics_path == (model / "topics.csv").resolve()


def test_run_from_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid = TimeGrid.regular(0.0, hop=5.0, window=5.0, n=2)
    block = FeatureBlock(values=np.zeros((2, 4), dtype=np.float32), grid=grid)
    runner = TopicModelRunner(num_topics=3, words_per_doc=1)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(
        block,
        tmp_path / "docs",
        tmp_path / "model",
        labels=np.array([0, 1]),
    )
    assert result.docs_path.is_file()
    assert result.docs_path.read_text().strip().splitlines() == ["0,0", "5000,1"]


def test_run_requires_docs(tmp_path: Path) -> None:
    runner = TopicModelRunner()
    with pytest.raises(FileNotFoundError):
        runner.run(tmp_path / "missing.csv", tmp_path / "model", vocab_size=1)


def test_run_from_block_linear_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from perchtopic.classify import build_model

    grid = TimeGrid.regular(0.0, hop=5.0, window=5.0, n=2)
    values = np.zeros((2, 4), dtype=np.float32)
    values[1] = 1.0
    block = FeatureBlock(values=values, grid=grid)
    probe = build_model(num_classes=2, embedding_dim=4)
    ckpt = tmp_path / "linear_model.pt"
    probe.save(ckpt, classes=["a", "b"])

    runner = TopicModelRunner(num_topics=2, words_per_doc=1)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(
        block,
        tmp_path / "docs",
        tmp_path / "model",
        linear_model=ckpt,
    )
    lines = result.docs_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert result.vocab_size == 2


def test_run_from_block_clusters_pcen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "stm.topicmodel.runner._cluster_word_ids",
        lambda values, n_clusters, random_state=0: (
            np.array([0, 1, 0, 1], dtype=np.int64),
            2,
        ),
    )
    grid = TimeGrid.regular(0.0, hop=0.5, window=0.5, n=4)
    block = FeatureBlock(
        values=np.arange(4 * 32, dtype=np.float32).reshape(4, 32),
        grid=grid,
        provenance={"extractor": "PcenExtractor"},
    )
    runner = TopicModelRunner(num_topics=3, words_per_doc=1)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(block, tmp_path / "docs", tmp_path / "model")
    assert result.docs_path.read_text().strip().splitlines() == [
        "0,0",
        "500,1",
        "1000,0",
        "1500,1",
    ]
    assert result.vocab_size == 2


def test_run_from_block_uses_precomputed_kmeans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("precomputed K-Means labels should not be clustered again")

    monkeypatch.setattr("stm.topicmodel.runner._cluster_word_ids", boom)
    grid = TimeGrid.regular(0.0, hop=0.5, window=0.5, n=4)
    block = FeatureBlock(
        values=np.array([[0], [1], [0], [1]], dtype=np.float32),
        grid=grid,
        provenance={
            "extractor": "PcenExtractor",
            "cluster": "kmeans",
            "n_clusters": 2,
        },
    )
    runner = TopicModelRunner(num_topics=3, words_per_doc=1)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(block, tmp_path / "docs", tmp_path / "model")
    assert result.docs_path.read_text().strip().splitlines() == [
        "0,0",
        "500,1",
        "1000,0",
        "1500,1",
    ]
    assert result.vocab_size == 2


def test_run_from_block_fuses_perch_and_pcen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_cluster(values, n_clusters, random_state=0):
        captured["shape"] = tuple(values.shape)
        return np.zeros(len(values), dtype=np.int64), 1

    monkeypatch.setattr("stm.topicmodel.runner._cluster_word_ids", fake_cluster)
    monkeypatch.setattr(
        "stm.topicmodel.runner._probe_word_ids",
        lambda values, linear_model: (np.ones(len(values), dtype=np.int64), 2),
    )
    grid = TimeGrid.regular(0.0, hop=1.0, window=1.0, n=3)
    perch = FeatureBlock(
        values=np.ones((3, 8), dtype=np.float32),
        grid=grid,
        provenance={"extractor": "Perch2Extractor"},
    )
    pcen = FeatureBlock(
        values=np.full((3, 4), 2.0, dtype=np.float32),
        grid=grid,
        provenance={"extractor": "PcenExtractor"},
    )
    runner = TopicModelRunner(num_topics=2, words_per_doc=1)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(
        [perch, pcen],
        tmp_path / "docs",
        tmp_path / "model",
        linear_model="probe",
    )
    assert captured["shape"] == (3, 4)
    assert result.docs_path.read_text().strip().splitlines() == [
        "0,1,2",
        "1000,1,2",
        "2000,1,2",
    ]


def test_run_from_block_pcen_ignores_mismatched_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from perchtopic.classify import build_model

    monkeypatch.setattr(
        "stm.topicmodel.runner._cluster_word_ids",
        lambda values, n_clusters, random_state=0: (
            np.array([2, 2], dtype=np.int64),
            3,
        ),
    )
    grid = TimeGrid.regular(0.0, hop=1.0, window=1.0, n=2)
    block = FeatureBlock(
        values=np.ones((2, 32), dtype=np.float32),
        grid=grid,
        provenance={"extractor": "PcenExtractor"},
    )
    probe = build_model(num_classes=2, embedding_dim=1536)
    runner = TopicModelRunner(num_topics=2, words_per_doc=1)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(
        block,
        tmp_path / "docs",
        tmp_path / "model",
        linear_model=probe,
    )
    assert result.docs_path.read_text().strip().splitlines() == ["0,2", "1000,2"]


def test_run_from_block_perch_clusters_without_linear_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_cluster(values, n_clusters, random_state=0):
        captured["shape"] = tuple(values.shape)
        captured["n_clusters"] = n_clusters
        return np.array([0, 1], dtype=np.int64), 2

    monkeypatch.setattr("stm.topicmodel.runner._cluster_word_ids", fake_cluster)
    grid = TimeGrid.regular(0.0, hop=1.0, window=1.0, n=2)
    block = FeatureBlock(
        values=np.ones((2, 8), dtype=np.float32),
        grid=grid,
        provenance={"extractor": "Perch2Extractor"},
    )
    runner = TopicModelRunner(num_topics=2, words_per_doc=1)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(block, tmp_path / "docs", tmp_path / "model")
    assert captured["shape"] == (2, 8)
    assert captured["n_clusters"] == 2
    assert result.docs_path.read_text().strip().splitlines() == ["0,0", "1000,1"]
    assert result.vocab_size == 2


def test_run_from_block_default_words_per_doc_sums_feature_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = TimeGrid.regular(0.0, hop=1.0, window=1.0, n=6)
    perch = FeatureBlock(
        values=np.zeros((6, 2), dtype=np.float32),
        grid=grid,
        provenance={"extractor": "Perch2Extractor"},
    )
    pcen = FeatureBlock(
        values=np.zeros((6, 3), dtype=np.float32),
        grid=grid,
        provenance={"extractor": "PcenExtractor"},
    )
    runner = TopicModelRunner(num_topics=2)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    result = runner.run_from_block(
        [perch, pcen],
        tmp_path / "docs",
        tmp_path / "model",
        labels=np.arange(6),
    )
    assert result.docs_path.read_text().strip().splitlines() == [
        "4000,0,1,2,3,4",
        "5000,5",
    ]


def test_run_from_block_perch_missing_probe_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = TimeGrid.regular(0.0, hop=1.0, window=1.0, n=2)
    block = FeatureBlock(
        values=np.ones((2, 8), dtype=np.float32),
        grid=grid,
        provenance={"extractor": "Perch2Extractor"},
    )
    runner = TopicModelRunner(num_topics=2)
    monkeypatch.setattr(runner, "_execute", lambda cmd, mount, log_path=None: "")
    with pytest.raises(FileNotFoundError):
        runner.run_from_block(
            block,
            tmp_path / "docs",
            tmp_path / "model",
            linear_model=tmp_path / "missing_linear_model.pt",
        )
