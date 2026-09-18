# stm, Apache-2.0 license
# Filename: raven_parser.py
# Description: parses annotation file created by the Raven software
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path

from stm.config import Config

LABEL_COLUMNS = ("label", "Label", "Classification", "classify", "Unit")
BACKGROUND_LABEL = "background"
BACKGROUND_KEEP_FRACTION = 0.10


def _row_label(row) -> str:
    for col in LABEL_COLUMNS:
        if col in row.index and not pd.isnull(row[col]):
            return str(row[col]).strip().replace("/", "_").replace("\\", "_")
    return "unlabeled"


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for lo, hi in ranges[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _gap_ranges(occupied: list[tuple[int, int]], duration: int) -> list[tuple[int, int]]:
    gaps = []
    t = 0
    for lo, hi in occupied:
        if lo > t:
            gaps.append((t, lo))
        t = max(t, hi)
    if t < duration:
        gaps.append((t, duration))
    return gaps


def _candidate_windows(gaps: list[tuple[int, int]], win: int) -> list[tuple[int, int]]:
    if win <= 0:
        return []
    windows = []
    for lo, hi in gaps:
        start = lo
        while start + win <= hi:
            windows.append((start, start + win))
            start += win
    return windows


def _sample_background_windows(
    occupied: list[tuple[int, int]],
    duration: int,
    win: int,
    fraction: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    candidates = _candidate_windows(_gap_ranges(_merge_ranges(occupied), duration), win)
    n_keep = int(round(len(candidates) * fraction))
    if n_keep <= 0:
        return []
    idx = rng.choice(len(candidates), size=min(n_keep, len(candidates)), replace=False)
    chosen = [candidates[int(i)] for i in np.atleast_1d(idx)]
    chosen.sort()
    return chosen


def _occupied_sample_ranges(df: pd.DataFrame, sampling_rate: int, max_samples: int) -> list[tuple[int, int]]:
    if df.empty or "Begin Time (s)" not in df.columns or "End Time (s)" not in df.columns:
        return []
    ranges = []
    for _, row in df.iterrows():
        start = max(0, int(row["Begin Time (s)"] * sampling_rate))
        stop = min(max_samples, int(row["End Time (s)"] * sampling_rate))
        if stop > start:
            ranges.append((start, stop))
    return ranges


def _build_df_unk(
    df: pd.DataFrame,
    prefix: str,
    max_samples: int,
    sampling_rate: int,
    window_seconds: int,
    config: Config,
    label_col: str | None,
) -> pd.DataFrame:
    """Random background windows that do not intersect ``df`` detections."""
    win = max(1, int(round(config.perch_audio_seconds * sampling_rate)))
    windows = _sample_background_windows(
        _occupied_sample_ranges(df, sampling_rate, max_samples),
        max_samples,
        win,
        BACKGROUND_KEEP_FRACTION,
        np.random.default_rng(),
    )
    col = label_col if label_col in LABEL_COLUMNS else "label"
    call_width = int(window_seconds * sampling_rate)
    selection0 = 0
    if not df.empty and "Selection" in df.columns:
        selection0 = int(df["Selection"].max())

    rows = []
    for i, (lo, hi) in enumerate(windows, 1):
        begin_s = lo / sampling_rate
        end_s = hi / sampling_rate
        middle = int(lo + ((hi - lo) / 2))
        selection = selection0 + i
        rows.append(
            {
                col: BACKGROUND_LABEL,
                "image_filename": (
                    f"{prefix}_{BACKGROUND_LABEL}.{lo}.{hi}.sel.{selection:02}.ch01.spectrogram.jpg"
                ),
                "Begin Time (s)": begin_s,
                "End Time (s)": end_s,
                "Selection": selection,
                "call_start": max(middle - call_width / 2, 0),
                "call_end": min(middle + call_width / 2, max_samples),
                "has_label": False,
            }
        )
    return pd.DataFrame(rows)


class RavenParser:
    """
    A class for parsing Raven Detections into pandas dataframes
    """

    def __init__(
        self,
        anno_path: Path,
        max_samples: int,
        sampling_rate: int,
        window_seconds: int,
        config: Config,
    ):
        """
        Process Raven file into pandas dataframe
        :param anno_path: path to text file with annotated call sounds
        :param sampling_rate: sampling rate of data detections are associated with
        :param window_seconds: window size in seconds for extracting detections; longer will be split into window_seconds segments
        :raises exception if no wav file associated with the annotation file or issue with parsing
        """
        self._num_detections = 0
        self._raven_file = anno_path.name
        self._df_unk = pd.DataFrame()

        print(f'Reading {anno_path}')
        self._df = pd.read_csv(anno_path.as_posix(), sep='\t')
        label_col = None

        if self._df.empty:
            print(f'Warning: {anno_path} has 0 detections')
        else:
            print(f'Found {len(self._df)} detections in {anno_path}')
            self._num_detections = len(self._df)

            call_width = int(window_seconds * sampling_rate)

            def call_start(start_time, end_time):
                start_samples = int(start_time * sampling_rate)
                end_samples = int(end_time * sampling_rate)
                middle = int(start_samples + ((end_samples - start_samples) / 2))
                return max(middle - call_width / 2, 0)

            def call_end(start_time, end_time):
                start_samples = int(start_time * sampling_rate)
                end_samples = int(end_time * sampling_rate)
                middle = int(start_samples + ((end_samples - start_samples) / 2))
                return min(middle + call_width / 2, max_samples)

            def has_label(call_label):
                if pd.isnull(call_label):
                    return True
                return False

            def image_filename(prefix, start_time, end_time, selection, call_label):
                start = int(start_time * sampling_rate)
                end = int(end_time * sampling_rate)
                if pd.isnull(call_label):
                    return f'{prefix}.{start}.{end}.sel.{int(selection):02}.ch01.spectrogram.jpg'
                else:
                    return f'{prefix}_{call_label}.{start}.{end}.sel.{int(selection):02}.ch01.spectrogram.jpg'

            self._df['call_start'] = self._df.apply(lambda x: call_start(x['Begin Time (s)'], x['End Time (s)']),
                                                    axis=1)
            self._df['call_end'] = self._df.apply(lambda x: call_end(x['Begin Time (s)'], x['End Time (s)']), axis=1)

            label_col = None
            for col in LABEL_COLUMNS:
                if col in self._df:
                    unique_labels = self._df[col].unique()
                    for label in unique_labels:
                        if pd.isnull(label):
                            continue

                    # assume only one labeled column in each file
                    label_col = col
                    break

            # Prefix to use when creating the segmented audio clips
            prefix = anno_path.stem

            if label_col is not None:
                self._df['image_filename'] = self._df.apply(
                    lambda x: image_filename(prefix, x['Begin Time (s)'], x['End Time (s)'],
                                             x['Selection'], x[label_col]), axis=1)
                self._df['has_label'] = self._df.apply(lambda x: has_label(x[label_col]), axis=1)
            else:
                self._df['image_filename'] = self._df.apply(
                    lambda x: image_filename(prefix, x['Begin Time (s)'], x['End Time (s)'],
                                             x['Selection'], np.nan), axis=1)
                self._df['has_label'] = self._df.apply(lambda x: False, axis=1)

            columns_keep = [label_col, 'image_filename', 'Begin Time (s)', 'End Time (s)', 'Selection', 'call_start', 'call_end', 'has_label']
            for c in self._df.columns:
                if c not in columns_keep:
                    self._df = self._df.drop(c, axis=1)

        self._df_unk = _build_df_unk(
            self._df,
            anno_path.stem,
            max_samples,
            sampling_rate,
            window_seconds,
            config,
            label_col,
        )
        print(f"Sampled {len(self._df_unk)} background windows")

    @property
    def num_detections(self):
        return self._num_detections

    @property
    def data(self):
        return self._df

    @property
    def data_unk(self):
        return self._df_unk

    @staticmethod
    def export_wavs(
        parser: "RavenParser",
        wav_path: Path,
        out_dir: Path,
        config: Config,
        include_unlabeled: bool = False,
    ) -> int:
        """Write WAV clips per detection, split and filled from ``config``.

        Intervals longer than ``Config.PERCH_TIME_BIN_SECONDS`` are cut into
        consecutive bins of that length. Each piece is then centered in a
        ``PERCH_TIME_BIN_SECONDS`` clip; short pieces use
        ``config.perch_window_fill`` (``tile`` repeats the audio; any other
        value zero-pads both sides).

        When ``include_unlabeled`` is true, also write ``parser._df_unk``
        background clips (non-overlapping times, ``perch_audio_seconds`` long).
        """
        frames = [parser.data] if parser.num_detections > 0 else []
        if include_unlabeled and not parser._df_unk.empty:
            frames.append(parser._df_unk)
        if not frames:
            print(f"No detections found for {wav_path}. Cannot export WAVs")
            return 0
        export_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        wav_path = Path(wav_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        info = sf.info(wav_path.as_posix())
        bin_samples = max(1, int(round(config.PERCH_TIME_BIN_SECONDS * info.samplerate)))
        n_written = _export_rows(export_df, wav_path, out_dir, info, bin_samples, config.perch_window_fill)

        print(f"Wrote {n_written} clips to {out_dir}/<label>/")
        return n_written


def _export_rows(df, wav_path: Path, out_dir: Path, info, bin_samples: int, fill: str) -> int:
    n_written = 0
    for _, row in df.iterrows():
        start = max(0, int(row["Begin Time (s)"] * info.samplerate))
        stop = min(info.frames, int(row["End Time (s)"] * info.samplerate))
        if stop <= start:
            continue

        slices = _bin_slices(start, stop, bin_samples)
        base_name = row["image_filename"].replace(".spectrogram.jpg", "")
        label_dir = out_dir / _row_label(row)
        label_dir.mkdir(parents=True, exist_ok=True)

        for i, (lo, hi) in enumerate(slices, 1):
            segment, _ = sf.read(wav_path.as_posix(), start=lo, stop=hi, always_2d=False)
            segment = _fill_centered(segment, bin_samples, fill)
            if len(slices) == 1:
                out_name = f"{base_name}.wav"
            else:
                out_name = f"{base_name}.p{i:02}.{lo}.{hi}.wav"
            sf.write((label_dir / out_name).as_posix(), segment, info.samplerate)
            n_written += 1
    return n_written


def _bin_slices(start: int, stop: int, bin_samples: int) -> list[tuple[int, int]]:
    """Non-overlapping ``bin_samples`` ranges covering ``[start, stop)``."""
    if stop - start <= bin_samples:
        return [(start, stop)]
    return [
        (lo, min(lo + bin_samples, stop))
        for lo in range(start, stop, bin_samples)
    ]


def _fill_centered(segment: np.ndarray, n: int, fill: str) -> np.ndarray:
    """Place ``segment`` in the middle of an ``n``-sample clip and fill the rest."""
    segment = np.asarray(segment)
    if segment.ndim > 1:
        return np.stack(
            [_fill_centered(segment[:, c], n, fill) for c in range(segment.shape[1])],
            axis=1,
        )

    length = int(segment.shape[0])
    if length >= n:
        extra = length - n
        lo = extra // 2
        return segment[lo : lo + n]

    if fill == "tile":
        if length == 0:
            return np.zeros(n, dtype=segment.dtype)
        start = (n - length) // 2
        return np.roll(np.resize(segment, n), start)

    out = np.zeros(n, dtype=segment.dtype)
    if length:
        start = (n - length) // 2
        out[start : start + length] = segment
    return out

