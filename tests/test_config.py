# stm, Apache-2.0 license
# Filename: test_config.py
# Description: Tests for Config YAML loading and PcenExtractor.from_config
from __future__ import annotations

from pathlib import Path

from perchtopic.config import Config
from perchtopic.features import PcenExtractor, Perch2Extractor


def test_config_load_reads_pcen_keys_and_ignores_extras(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "rost_path: ./rost-cli/bin/",
                "num_topics: 23",
                "window_size: 1024",
                "overlap: 0.25",
                "num_mel_bins: 64",
                "sigma: 2.0",
                "pcen_fmin: 10.0",
                "pcen_fmax: 4000.0",
                "pcen_gain: 0.8",
                "pcen_bias: 3.0",
                "pcen_time_constant: 0.2",
                "pcen_power_to_db: false",
            ]
        )
        + "\n"
    )
    config = Config.load(
        yaml_path,
        wav_path=tmp_path / "audio",
        output_path=tmp_path / "out",
    )
    assert config.window_size == 1024
    assert config.overlap == 0.25
    assert config.num_mel_bins == 64
    assert config.sigma == 2.0
    assert config.pcen_fmin == 10.0
    assert config.pcen_fmax == 4000.0
    assert config.pcen_gain == 0.8
    assert config.pcen_bias == 3.0
    assert config.pcen_time_constant == 0.2
    assert config.pcen_power_to_db is False
    assert config.wav_path == tmp_path / "audio"
    assert config.doc_path == tmp_path / "out" / "doc"


def test_pcen_extractor_from_config(tmp_path: Path) -> None:
    config = Config(
        wav_path=tmp_path,
        output_path=tmp_path / "out",
        window_size=1024,
        overlap=0.25,
        num_mel_bins=64,
        sigma=None,
        pcen_fmin=50.0,
        pcen_fmax=4000.0,
        pcen_gain=0.7,
        pcen_bias=4.0,
        pcen_time_constant=0.3,
        pcen_power_to_db=False,
    )
    extractor = PcenExtractor.from_config(config, timeout=12.0)
    assert extractor.timeout == 12.0
    assert extractor.window_size == 1024
    assert extractor.overlap == 0.25
    assert extractor.num_mel_bins == 64
    assert extractor.sigma is None
    assert extractor.fmin == 50.0
    assert extractor.fmax == 4000.0
    assert extractor.gain == 0.7
    assert extractor.bias == 4.0
    assert extractor.time_constant == 0.3
    assert extractor.power_to_db is False
    assert extractor.cluster is None
    assert extractor.n_clusters is None


def test_perch_extractor_from_config(tmp_path: Path) -> None:
    model = tmp_path / "perch_v2.onnx"
    model.write_bytes(b"")
    config = Config(
        wav_path=tmp_path,
        output_path=tmp_path / "out",
        perch_window_fill="tile",
    )
    extractor = Perch2Extractor.from_config(config, model=model, timeout=9.0)
    assert extractor.window_fill == "tile"
    assert extractor.output_path == tmp_path / "out"
    assert extractor.timeout == 9.0
    assert extractor.cache is True
    assert extractor.model == model


def test_config_load_packaged_yaml(tmp_path: Path) -> None:
    packaged = Path(__file__).resolve().parents[1] / "stm" / "config.yaml"
    config = Config.load(packaged, wav_path=tmp_path, output_path=tmp_path / "out")
    extractor = PcenExtractor.from_config(config)
    assert extractor.window_size == 512
    assert extractor.num_mel_bins == 32
    assert extractor.gain == 0.5
    assert extractor.fmax == 8000.0
    assert extractor.sigma == 1.0
    assert extractor.cluster is None
    assert extractor.n_clusters == 24


def test_pcen_extractor_from_config_kmeans(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "window_size: 512",
                "pcen_cluster: kmeans",
                "pcen_n_clusters: 8",
            ]
        )
        + "\n"
    )
    config = Config.load(yaml_path, wav_path=tmp_path, output_path=tmp_path / "out")
    extractor = PcenExtractor.from_config(config)
    assert extractor.cluster == "kmeans"
    assert extractor.n_clusters == 8
    assert extractor.embedding_dim == 1


def test_pcen_extractor_from_config_kmeans_knee(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "window_size: 512",
                "pcen_cluster: kmeans",
                "pcen_n_clusters: null",
            ]
        )
        + "\n"
    )
    config = Config.load(yaml_path, wav_path=tmp_path, output_path=tmp_path / "out")
    extractor = PcenExtractor.from_config(config)
    assert extractor.cluster == "kmeans"
    assert extractor.n_clusters is None
