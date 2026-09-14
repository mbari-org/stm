# stm, Apache-2.0 license
# Filename: plotter.py
# Description: Spectrogram and topic plots from a trained ROST model
from __future__ import annotations

import colorsys
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy import signal as scipy_signal
from scipy.optimize import linear_sum_assignment

from perchtopic.raven_parser import LABEL_COLUMNS

BACKGROUND_BAR_COLOR = "white"
WINDOW_SIZE = 512
OVERLAP = 0.5
FREQ_SUBSET = (0.0, 8000.0)
THETA_NAME = "theta.csv"
META_NAME = "model_meta.json"


@dataclass
class PlotResult:
    """Figures produced by :meth:`Plotter.plot`, and where they were saved.

    ``paths`` is empty when ``save=False``. Figures are returned whether or
    not they were saved or shown, so callers can compose or re-save them.
    """

    figures: list[Figure] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.figures)

    def __iter__(self):
        return iter(self.figures)


class Plotter:
    """Spectrogram + topic (+ ground-truth) plots from a trained ROST model.

    *model_dir* is the directory written by :class:`TopicModelRunner` (must
    contain ``theta.csv``). Chunking matches perch2topic ``visualize_all.py``:
    overlapping windows of *chunk_size* seconds, *overlap_fraction* overlap.

    One row of ``theta`` is one document. *document_seconds* is how much audio
    a document spans; it is read from ``model_meta.json`` when the run wrote
    one, so :meth:`plot` usually needs no timing arguments. Pass *config* to
    take spectrogram settings (``window_size``, ``overlap``, ``pcen_fmin`` /
    ``pcen_fmax``) from the same :class:`~stm.config.Config` used to
    train, instead of this module's defaults.
    """

    def __init__(
        self,
        model_dir: Path | str,
        config=None,
        document_seconds: float | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self._verify_model()
        self.theta = pd.read_csv(self.theta_path, header=None).values
        if self.theta.ndim != 2 or self.theta.shape[0] < 1:
            raise ValueError(f"{self.theta_path} has no document-topic rows")
        self.config = config
        self.meta = self._read_meta()
        if document_seconds is None:
            document_seconds = self.meta.get("document_seconds")
        self.document_seconds = (
            None if document_seconds is None else float(document_seconds)
        )
        self._spectrogram_cache: dict[tuple, tuple] = {}

    @classmethod
    def from_result(cls, result, config=None) -> "Plotter":
        """Build a Plotter from a :class:`TopicModelResult`.

        Takes the model directory and document duration straight off the run,
        so the plot's time axis cannot disagree with the model it describes.
        """
        return cls(
            result.out_dir,
            config=config,
            document_seconds=getattr(result, "document_seconds", None),
        )

    @property
    def theta_path(self) -> Path:
        return self.model_dir / THETA_NAME

    @property
    def meta_path(self) -> Path:
        return self.model_dir / META_NAME

    @property
    def num_documents(self) -> int:
        return int(self.theta.shape[0])

    @property
    def num_topics(self) -> int:
        return int(self.theta.shape[1])

    def _read_meta(self) -> dict:
        if not self.meta_path.is_file():
            return {}
        try:
            data = json.loads(self.meta_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"Ignoring unreadable {self.meta_path}: {exc}")
            return {}
        return data if isinstance(data, dict) else {}

    def _verify_model(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model directory does not exist: {self.model_dir}")
        if not self.model_dir.is_dir():
            raise NotADirectoryError(f"model path is not a directory: {self.model_dir}")
        if not self.theta_path.is_file():
            raise FileNotFoundError(
                f"no topic model in {self.model_dir}: missing {THETA_NAME}. "
                "TopicModelRunner.run() only writes phi/theta when ROST produced "
                "topicmodel.csv and topics.hist.csv and the topic count is known, "
                "so the run may have completed without them."
            )

    def plot(
        self,
        source: Path | str,
        *,
        document_seconds: float | None = None,
        selections: Path | str | None = None,
        chunk_size: float = 60.0,
        overlap_fraction: float = 0.25,
        n_chunks: int = 2,
        show: bool = False,
        save: bool = True,
        window_size: int | None = None,
        overlap: float | None = None,
        freq_range: tuple[float, float] | None = None,
    ) -> PlotResult:
        """Plot the first *n_chunks* overlapping time windows of *source*.

        *source* is a WAV path or a directory of WAVs. *document_seconds* is
        how much audio one ``theta`` row spans; when omitted it comes from
        ``model_meta.json`` (or from :meth:`from_result`). *selections* may be
        a Raven file or a directory to look one up in; by default a sibling
        ``<wav stem>.selections.txt`` is used when present.

        *window_size*, *overlap* and *freq_range* override the spectrogram
        settings, which otherwise come from *config* if one was given.
        Returns a :class:`PlotResult` holding the figures and any saved paths.
        """
        if n_chunks < 1:
            raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if not 0.0 <= overlap_fraction < 1.0:
            raise ValueError(
                f"overlap_fraction must be in [0, 1), got {overlap_fraction}"
            )

        starts, ends, secs_per_doc = self._document_times(document_seconds)
        theta = self.theta[: len(starts)]
        wav = _resolve_wav(source)
        modeled_end = float(ends[-1])
        chunk_times = _chunk_windows(
            modeled_end, chunk_size, overlap_fraction, n_chunks
        )
        if not chunk_times:
            raise ValueError(
                "no time chunks to plot; check document_seconds and chunk_size"
            )
        print(
            f"{wav.name}: {self.num_documents} documents x {secs_per_doc:g}s "
            f"= {modeled_end:.1f}s modelled, {self.num_topics} topics"
        )
        if len(chunk_times) < n_chunks:
            print(
                f"  {n_chunks} chunks requested, {len(chunk_times)} fit in "
                f"{modeled_end:.1f}s"
            )
        print(f"  chunks: {chunk_times}")

        gt_labels, gt_legend = self._ground_truth(selections, wav, starts, ends)

        frequencies, display_times, display_stft = self._spectrogram_for(
            wav,
            max(end for _, end in chunk_times),
            *self._spectrogram_settings(window_size, overlap, freq_range),
        )
        topic_colors, gt_colors, gt_legend_with_topics, num_classes = _colors_for_topics(
            theta, gt_labels, gt_legend
        )

        result = PlotResult()
        for start, end in chunk_times:
            print(f"Processing {wav}: chunk from {start} to {end} seconds")
            start_doc = math.floor(start / secs_per_doc)
            end_doc = math.ceil(end / secs_per_doc)
            theta_chunk = theta[start_doc:end_doc]
            doc_starts = (np.arange(len(theta_chunk)) + start_doc) * secs_per_doc
            start_frame = int(np.searchsorted(display_times, start, side="left"))
            end_frame = int(np.searchsorted(display_times, end, side="right"))
            prefix = f"{wav.stem}_{start:06.0f}_{end:06.0f}_topics_"
            fig = _plot_topic_chunk(
                display_stft[:, start_frame:end_frame],
                frequencies,
                display_times[start_frame:end_frame],
                theta_chunk,
                doc_starts,
                secs_per_doc,
                topic_colors,
                prefix,
                gt_chunk=None if gt_labels is None else gt_labels[start_doc:end_doc],
                gt_legend_with_topics=gt_legend_with_topics,
                gt_colors=gt_colors,
                num_classes=num_classes,
            )
            if save:
                dest = self.model_dir / f"{prefix}.png"
                fig.savefig(dest, dpi=150, bbox_inches="tight")
                result.paths.append(dest)
            if show:
                plt.show()
            result.figures.append(fig)
        return result

    def _document_times(
        self, document_seconds: float | None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """One interval per ``theta`` row, tiled at *document_seconds*."""
        seconds = self.document_seconds if document_seconds is None else document_seconds
        if seconds is None:
            raise ValueError(
                "document_seconds is required and could not be discovered: no "
                f"{META_NAME} in {self.model_dir}. Pass document_seconds=, or "
                "build the Plotter with Plotter.from_result(result)."
            )
        seconds = float(seconds)
        if seconds <= 0.0:
            raise ValueError(f"document_seconds must be > 0, got {seconds}")
        starts = np.arange(self.num_documents, dtype=float) * seconds
        return starts, starts + seconds, seconds

    def _spectrogram_settings(
        self,
        window_size: int | None,
        overlap: float | None,
        freq_range: tuple[float, float] | None,
    ) -> tuple[int, float, tuple[float, float]]:
        """Explicit arguments win, then *config*, then module defaults."""
        cfg = self.config
        if window_size is None:
            window_size = getattr(cfg, "window_size", None) or WINDOW_SIZE
        if overlap is None:
            overlap = getattr(cfg, "overlap", None)
            overlap = OVERLAP if overlap is None else overlap
        if freq_range is None:
            lo = getattr(cfg, "pcen_fmin", None)
            hi = getattr(cfg, "pcen_fmax", None)
            freq_range = (
                FREQ_SUBSET[0] if lo is None else float(lo),
                FREQ_SUBSET[1] if hi is None else float(hi),
            )
        window_size = int(window_size)
        overlap = float(overlap)
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        lo, hi = float(freq_range[0]), float(freq_range[1])
        if hi <= lo:
            raise ValueError(f"freq_range must be (low, high) with high > low, got {freq_range}")
        return window_size, overlap, (lo, hi)

    def _spectrogram_for(
        self,
        wav: Path,
        max_t: float,
        window_size: int,
        overlap: float,
        freq_range: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`_load_spectrogram` memoised, so repeat plots don't re-read the WAV."""
        key = (str(wav), round(float(max_t), 3), window_size, overlap, freq_range)
        if key not in self._spectrogram_cache:
            self._spectrogram_cache[key] = _load_spectrogram(
                wav, max_t, window_size, overlap, freq_range
            )
        return self._spectrogram_cache[key]

    def _ground_truth(
        self,
        selections: Path | str | None,
        wav: Path,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> tuple[np.ndarray | None, list[str] | None]:
        """Resolve and load Raven ground truth, reporting what was found."""
        default_name = f"{wav.stem}.selections.txt"
        if selections is None:
            sel_path = wav.with_name(default_name)
            explicit = False
        else:
            sel_path = Path(selections)
            if sel_path.is_dir():
                sel_path = sel_path / default_name
            explicit = True

        if not sel_path.is_file():
            message = f"  no ground truth: {sel_path} not found"
            if explicit:
                raise FileNotFoundError(message.strip())
            print(message)
            return None, None

        gt_labels, gt_legend = _gt_labels_from_raven(sel_path, starts, ends)
        n_hit = int(np.count_nonzero(gt_labels > 0))
        print(
            f"  ground truth: {sel_path.name}, {len(gt_legend) - 1} class(es), "
            f"{n_hit}/{len(gt_labels)} documents labelled"
        )
        if n_hit == 0:
            print("  (no document overlaps any selection; check the time units)")
        return gt_labels, gt_legend


def _resolve_wav(source: Path | str) -> Path:
    source = Path(source)
    if source.is_dir():
        wavs = sorted(p for p in source.rglob("*.wav") if p.is_file())
        if not wavs:
            raise FileNotFoundError(f"No wav clips under {source}")
        return wavs[0]
    if source.suffix.lower() != ".wav" and source.with_suffix(".wav").is_file():
        return source.with_suffix(".wav")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _chunk_windows(
    modeled_end: float,
    chunk_size: float,
    overlap_fraction: float,
    n_chunks: int,
) -> list[tuple[float, float]]:
    step = max(1.0, chunk_size * (1.0 - overlap_fraction))
    times = [
        (float(i), min(float(i) + chunk_size, modeled_end))
        for i in np.arange(0.0, modeled_end, step)
    ]
    return times[:n_chunks]


def _load_spectrogram(
    wav: Path,
    max_t: float,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
    freq_range: tuple[float, float] = FREQ_SUBSET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio, fs = sf.read(wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio[: int(min(len(audio), math.ceil(max_t * fs) + window_size))]
    hop_length = int(window_size * (1 - overlap))
    frequencies, display_times, spec = scipy_signal.spectrogram(
        audio,
        fs=fs,
        nperseg=window_size,
        noverlap=max(0, window_size - hop_length),
        scaling="spectrum",
        mode="magnitude",
    )
    display_stft = 20 * np.log10(np.maximum(spec, 1e-10))
    freq_lo = np.searchsorted(frequencies, freq_range[0], side="left")
    freq_hi = np.searchsorted(frequencies, freq_range[1], side="right")
    return frequencies[freq_lo:freq_hi], display_times, display_stft[freq_lo:freq_hi]


def _gt_labels_from_raven(
    sel_path: Path, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    sel = pd.read_csv(sel_path, sep="\t")
    label_col = next((c for c in LABEL_COLUMNS if c in sel.columns), None)
    if label_col is None:
        raise KeyError(
            f"{sel_path} has no label column; expected one of {list(LABEL_COLUMNS)}, "
            f"got {list(sel.columns)}"
        )
    labels = np.zeros(len(starts), dtype=int)
    unit_to_id: dict[str, int] = {}
    next_id = 1
    for _, row in sel.iterrows():
        if pd.isnull(row[label_col]):
            continue
        unit = str(row[label_col])
        if unit not in unit_to_id:
            unit_to_id[unit] = next_id
            next_id += 1
        begin, finish = float(row["Begin Time (s)"]), float(row["End Time (s)"])
        overlap = np.minimum(ends, finish) - np.maximum(starts, begin)
        labels[overlap > 0] = unit_to_id[unit]
    legend = ["background"] + [None] * (next_id - 1)
    for name, idx in unit_to_id.items():
        legend[idx] = name
    return labels, legend


def _colors_for_topics(
    theta: np.ndarray,
    gt_labels: np.ndarray | None,
    gt_legend: list[str] | None,
) -> tuple[list | None, list | None, list[str] | None, int | None]:
    n_topics = theta.shape[1]
    topic_colors = _topic_color_list(n_topics) if n_topics else None
    if gt_labels is None or gt_legend is None:
        return topic_colors, None, None, None
    gt_to_topic = _match_gt_classes_to_topics(gt_labels, theta)
    num_classes = max(int(gt_labels.max()) + 1, len(gt_legend))
    gt_colors = [
        _label_hash_color(gt_legend[g] if g < len(gt_legend) else f"GT{g}")
        for g in range(num_classes)
    ]
    if num_classes:
        gt_colors[0] = BACKGROUND_BAR_COLOR
    claimed: set[int] = set()
    for g in range(1, num_classes):
        t = int(gt_to_topic[g])
        if topic_colors is not None and 0 <= t < len(topic_colors) and t not in claimed:
            topic_colors[t] = gt_colors[g]
            claimed.add(t)
    gt_legend_with_topics = [
        f"{gt_legend[g]}→T{int(gt_to_topic[g])}" for g in range(num_classes)
    ]
    return topic_colors, gt_colors, gt_legend_with_topics, num_classes


def _format_hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def _apply_time_axis(ax, tick_seconds: float) -> None:
    formatter = FuncFormatter(lambda x, pos: _format_hhmmss(x))
    ax.xaxis.set_minor_locator(MultipleLocator(tick_seconds))
    ax.xaxis.set_minor_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_xticklabels(minor=True), rotation=45, ha="right")


def _spectrogram(stft, frequencies, display_times, ax=None):
    ax = plt.gca() if ax is None else ax
    num_frames = stft.shape[1]
    freq_min = float(frequencies.min()) if len(frequencies) else 0.0
    freq_max = float(frequencies.max()) if len(frequencies) else 1.0
    dt = float(np.median(np.diff(display_times))) if num_frames > 1 else 1.0
    t0 = float(display_times[0]) if len(display_times) else 0.0
    t1 = float(display_times[-1]) if len(display_times) else t0
    ax.imshow(
        stft,
        origin="lower",
        aspect="auto",
        extent=[t0, t1 + dt, freq_min, freq_max],
        cmap="Blues",
    )
    ax.set_ylabel("Frequency (Hz)")
    return ax, (t0, t1 + dt)


def _stacked_bar(
    data,
    legend=None,
    legend_labels=None,
    colors=None,
    x=None,
    bar_width=1.0,
    ax=None,
) -> None:
    if len(data) == 0:
        return
    ax = plt.gca() if ax is None else ax
    data_height = len(data[0])
    patches = []
    if x is None:
        x = np.arange(len(data), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    bottom = np.zeros(len(data))
    for z in range(data_height):
        col = [row[z] for row in data]
        color = None if colors is None else colors[z % len(colors)]
        patches.append(
            ax.bar(
                x=x, height=col, width=bar_width, bottom=bottom, align="edge", color=color
            )
        )
        bottom = np.add(col, bottom)
    if legend_labels is None:
        legend_labels = [f"{legend}{i}" for i in range(data_height)]
    # Outside the axes, so many topics never cover the bars.
    ax.legend(
        tuple(_[0] for _ in patches),
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        ncol=max(1, math.ceil(data_height / 16)),
        fontsize="small",
        frameon=False,
    )


def _labels_to_one_hot(labels, num_classes=None):
    labels = np.asarray(labels).flatten().astype(int)
    if labels.size == 0:
        return []
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    one_hot = np.zeros((len(labels), num_classes), dtype=float)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot.tolist()


def _match_gt_classes_to_topics(gt_labels, theta):
    gt_labels = np.asarray(gt_labels).flatten().astype(int)
    theta = np.asarray(theta, dtype=float)
    if gt_labels.size == 0 or theta.size == 0:
        return np.zeros(0, dtype=int)
    n_topics = theta.shape[1]
    num_classes = int(gt_labels.max()) + 1
    overlap = np.zeros((num_classes, n_topics), dtype=float)
    for g in range(num_classes):
        mask = gt_labels == g
        if np.any(mask):
            overlap[g] = theta[mask].sum(axis=0)
    gt_to_topic = np.zeros(num_classes, dtype=int)
    if num_classes and n_topics:
        row_ind, col_ind = linear_sum_assignment(-overlap)
        for row, col in zip(row_ind, col_ind):
            gt_to_topic[int(row)] = int(col)
    return gt_to_topic


def _label_hash_color(label, saturation=0.60, value=0.85):
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    return colorsys.hsv_to_rgb(hue, saturation, value)


_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def _topic_color_list(n_topics):
    """Distinct colours for any topic count.

    ``tab20`` up to 20 topics, then golden-ratio hue spacing, which stays
    well separated for arbitrary *n* instead of repeating every 20.
    """
    n_topics = int(n_topics)
    if n_topics <= 0:
        return []
    if n_topics <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n_topics)]
    return [
        colorsys.hsv_to_rgb((i * _GOLDEN_RATIO_CONJUGATE) % 1.0, 0.65, 0.90)
        for i in range(n_topics)
    ]


def _plot_topic_chunk(
    stft,
    frequencies,
    display_times,
    theta_chunk,
    doc_starts,
    secs_per_doc,
    topic_colors,
    prefix,
    gt_chunk=None,
    gt_legend_with_topics=None,
    gt_colors=None,
    num_classes=None,
):
    n_rows = 3 if gt_chunk is not None and len(gt_chunk) else 2
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 4 * n_rows), sharex=True, layout="constrained"
    )
    if n_rows == 1:
        axes = [axes]
    spec_ax, time_xlim = _spectrogram(stft, frequencies, display_times, ax=axes[0])
    spec_ax.set_title(prefix)
    _stacked_bar(
        theta_chunk,
        legend="T",
        colors=topic_colors,
        x=doc_starts,
        bar_width=secs_per_doc,
        ax=axes[1],
    )
    axes[1].set_ylabel("Topic Probability")
    if n_rows == 3:
        _stacked_bar(
            _labels_to_one_hot(gt_chunk, num_classes=num_classes),
            legend="GT",
            legend_labels=gt_legend_with_topics,
            colors=gt_colors,
            x=doc_starts,
            bar_width=secs_per_doc,
            ax=axes[2],
        )
        axes[2].set_ylabel("Ground Truth")
    axes[-1].set_xlim(*time_xlim)
    tick_seconds = (time_xlim[1] - time_xlim[0]) / 10
    _apply_time_axis(axes[-1], tick_seconds)
    axes[-1].set_xlabel("Time")
    for ax in axes:
        ax.label_outer()
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.set_xlabel("")
    return fig