# stm, Apache-2.0 license
# Filename: config.py
# Description: Configuration parsing
from pathlib import Path
from dataclasses import dataclass, field, fields

this_dir = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    PERCH_TIME_BIN_SECONDS = 5

    wav_path: Path = this_dir / "notebooks" / "dataset"
    output_path: Path = this_dir / "notebooks" / "output"
    perch_hop_seconds: float = 0.5
    perch_audio_seconds: float = 2.0
    perch_window_fill: str = field(default="fill")
    window_size: int = 512
    overlap: float = 0.5
    num_mel_bins: int = 32
    sigma: float | None = 1.0
    pcen_fmin: float = 0.0
    pcen_fmax: float = 8000.0
    pcen_gain: float = 0.5
    pcen_bias: float = 2.0
    pcen_time_constant: float = 0.4
    alpha: float = 0.01
    beta: float = 0.1
    gamma: float = 0.001
    word_per_doc: int = 32
    max_topic: bool = True  # When True, export a single max topic per time period in topics.maxlikelihood_with_time.csv.
    g_time: int = 1  # Depth of temporal neighborhood in cells
    cell_space: int = 0  # cell width in space dim
    threads: int = 64
    online: bool = False
    online_mint: int = 5
    pcen_power_to_db: bool = True
    pcen_cluster: str | None = None
    pcen_n_clusters: int | None = None
    num_topics: int | None = None

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        wav_path: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> "Config":
        """Load PCEN/Perch defaults from a YAML mapping.

        Extra YAML keys are ignored. *wav_path* and *output_path* override
        (or supply) those fields when they are missing from the file.

        Example::

            from pathlib import Path
            from stm.config import Config

            config = Config.load(
                Path("../config.yaml"),
                wav_path=Path("dataset"),
                output_path=Path("output"),
            )
            config.verify()
        """
        import yaml

        path = Path(path)
        with path.open() as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        names = {item.name for item in fields(cls) if item.init}
        kwargs = {key: value for key, value in raw.items() if key in names}
        if wav_path is not None:
            kwargs["wav_path"] = wav_path
        if output_path is not None:
            kwargs["output_path"] = output_path
        if "wav_path" not in kwargs:
            raise ValueError("Config.load requires wav_path in the YAML or as wav_path=")
        if "output_path" not in kwargs:
            raise ValueError(
                "Config.load requires output_path in the YAML or as output_path="
            )
        kwargs["wav_path"] = Path(kwargs["wav_path"])
        kwargs["output_path"] = Path(kwargs["output_path"])
        return cls(**kwargs)

    @property
    def model_path(self) -> Path:
        return self.output_path / "model"

    @property
    def doc_path(self) -> Path:
        return self.output_path / "doc"

    def verify(self):
        wav_path_exists = Path(self.wav_path).exists()
        output_path_exists = Path(self.output_path).exists()

        print(
            f"\n==================="
            f"\nCONFIG"
            f"\n==================="
            f"\nwav_path: {self.wav_path} "
            f"\noutput_path: {self.doc_path}"
            f"\nword_per_doc: {self.word_per_doc}"
            f"\nmax_topic: {self.max_topic}"
            f"\ng_time: {self.g_time}"
            f"\ncell_space: {self.cell_space}"
            f"\nthreads: {self.threads}"
            f"\nonline: {self.online}"
            f"\nonline_mint: {self.online_mint}"
            f"\nmodel_path: {self.model_path}"
            f"\nperch_hop_seconds: {self.perch_hop_seconds}"
            f"\nperch_audio_seconds: {self.perch_audio_seconds}"
            f"\nperch_window_fill: {self.perch_window_fill}"
            f"\nPERCH_TIME_BIN_SECONDS: {self.PERCH_TIME_BIN_SECONDS}"
            f"\nwindow_size: {self.window_size}"
            f"\noverlap: {self.overlap}"
            f"\nnum_mel_bins: {self.num_mel_bins}"
            f"\nsigma: {self.sigma}"
            f"\npcen_fmin: {self.pcen_fmin}"
            f"\npcen_fmax: {self.pcen_fmax}"
            f"\npcen_gain: {self.pcen_gain}"
            f"\npcen_bias: {self.pcen_bias}"
            f"\npcen_time_constant: {self.pcen_time_constant}"
            f"\npcen_power_to_db: {self.pcen_power_to_db}"
            f"\npcen_cluster: {self.pcen_cluster}"
            f"\npcen_n_clusters: {self.pcen_n_clusters}"
        )

        if not wav_path_exists:
            print(
                f"\n==================="
                f"\nERROR: wav_path {self.wav_path} not exists"
                f"\n==================="
            )

        if not output_path_exists:
            print(
                f"\n==================="
                f"\nERROR: output_path {self.output_path} not exists"
                f"\n==================="
            )


def main() -> None:
    config = Config()
    config.verify()



if __name__ == "__main__":
    main()
