"""
Generate audio with continuous tonals at 30 Hz and 100 Hz for a given duration,
with silence between the tonal segments.
"""

import numpy as np
from scipy.io import wavfile


def generate_tonal(
    freq_hz: float,
    duration_sec: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a continuous tonal (sine wave) at the given frequency.

    Parameters
    ----------
    freq_hz : float
        Frequency of the tone in Hz.
    duration_sec : float
        Duration of the tonal in seconds.
    sample_rate : int
        Sample rate in Hz (default 44100).
    amplitude : float
        Amplitude in [0, 1] (default 0.5).

    Returns
    -------
    np.ndarray
        Mono float array in [-1, 1].
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    segment = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return segment.astype(np.float32)


def generate_silence(duration_sec: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a silent segment."""
    n_samples = int(duration_sec * sample_rate)
    return np.zeros(n_samples, dtype=np.float32)


def generate_tonals_audio(
    total_duration_sec: float = 30.0,
    segment_duration_sec: float = 8.0,
    silence_between_sec: float = 3.0,
    freqs_hz: tuple[float, float] = (30.0, 100.0),
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate full audio: 30 Hz tonal, silence, 100 Hz tonal, silence, etc.
    Repeats pattern to fill total_duration_sec. Each tonal is continuous (no pulsing).

    Parameters
    ----------
    total_duration_sec : float
        Total duration in seconds (default 30).
    segment_duration_sec : float
        Duration of each continuous tonal segment in seconds (default 8).
    silence_between_sec : float
        Duration of silence between tonal segments in seconds (default 3).
    freqs_hz : tuple
        (freq_low, freq_high) in Hz (default (30, 100)).
    sample_rate : int
        Sample rate (default 44100).
    amplitude : float
        Amplitude in [0, 1] (default 0.5).

    Returns
    -------
    np.ndarray
        Mono float array in [-1, 1].
    """
    f_low, f_high = freqs_hz
    segments = []
    elapsed = 0.0

    while elapsed < total_duration_sec:
        # 30 Hz tonal
        seg_low = generate_tonal(
            f_low,
            segment_duration_sec,
            sample_rate=sample_rate,
            amplitude=amplitude,
        )
        segments.append(seg_low)
        elapsed += segment_duration_sec
        if elapsed >= total_duration_sec:
            break

        # Silence
        seg_silence = generate_silence(silence_between_sec, sample_rate)
        segments.append(seg_silence)
        elapsed += silence_between_sec
        if elapsed >= total_duration_sec:
            break

        # 100 Hz tonal
        seg_high = generate_tonal(
            f_high,
            segment_duration_sec,
            sample_rate=sample_rate,
            amplitude=amplitude,
        )
        segments.append(seg_high)
        elapsed += segment_duration_sec
        if elapsed >= total_duration_sec:
            break

        # Silence before next cycle
        seg_silence = generate_silence(silence_between_sec, sample_rate)
        segments.append(seg_silence)
        elapsed += silence_between_sec

    audio = np.concatenate(segments)
    # Trim to exact total duration
    n_total = int(total_duration_sec * sample_rate)
    return audio[:n_total]


def generate_and_save(
    output_path: str,
    total_duration_sec: float = 30.0,
    segment_duration_sec: float = 8.0,
    silence_between_sec: float = 3.0,
    freqs_hz: tuple[float, float] = (30.0, 100.0),
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate tonal audio and save to a file.

    Parameters
    ----------
    output_path : str
        Path where to save the audio (WAV file).
    total_duration_sec : float
        Total duration in seconds (default 30).
    segment_duration_sec : float
        Duration of each continuous tonal segment in seconds (default 8).
    silence_between_sec : float
        Silence between segments in seconds (default 3).
    freqs_hz : tuple
        (freq_low, freq_high) in Hz (default (30, 100)).
    sample_rate : int
        Sample rate (default 44100).
    amplitude : float
        Amplitude in [0, 1] (default 0.5).

    Returns
    -------
    np.ndarray
        The generated audio (float, mono).
    """
    audio = generate_tonals_audio(
        total_duration_sec=total_duration_sec,
        segment_duration_sec=segment_duration_sec,
        silence_between_sec=silence_between_sec,
        freqs_hz=freqs_hz,
        sample_rate=sample_rate,
        amplitude=amplitude,
    )
    save_wav(output_path, audio, sample_rate)
    return audio


def save_wav(path: str, audio: np.ndarray, sample_rate: int = 44100) -> None:
    """
    Save float audio [-1, 1] to a WAV file (16-bit).

    Parameters
    ----------
    path : str
        Output file path.
    audio : np.ndarray
        Float array in [-1, 1].
    sample_rate : int
        Sample rate in Hz.
    """
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio_int16)


def main() -> None:
    """Generate 30 s of 30 Hz and 100 Hz continuous tonals with silence between them."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate tonal audio (continuous sine segments)")
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        default="tonals_30s.wav",
        metavar="PATH",
        help="Path where to save the audio file (default: tonals_30s.wav)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=30.0,
        help="Total duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=8.0,
        help="Duration of each 30 Hz / 100 Hz tonal segment in seconds (default: 8)",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=3.0,
        help="Silence between segments in seconds (default: 3)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate (default: 44100)",
    )
    args = parser.parse_args()

    audio = generate_tonals_audio(
        total_duration_sec=args.duration,
        segment_duration_sec=args.segment_duration,
        silence_between_sec=args.silence,
        sample_rate=args.sr,
    )
    save_wav(args.output_path, audio, args.sr)
    print(f"Saved {args.duration} s to {args.output_path} ({args.sr} Hz)")


if __name__ == "__main__":
    main()
