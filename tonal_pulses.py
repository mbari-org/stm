#!/usr/bin/env python3
"""
Generate tonal pulse datasets using SOX via subprocess.
"""

import subprocess
from pathlib import Path


SR = 16000
FREQ_HZ = 1000
SOX_PATH = "/opt/homebrew/bin/sox" # Change this to your SOX path


def run_sox(
    output_path: Path,
    total_duration: float,
    tone_duration: float,
    silence_duration: float,
    freq_hz: float = FREQ_HZ,
    sample_rate: int = SR,
) -> None:
    """
    Generate tone/silence pulses using SOX.

    Pattern:
        tone (tone_duration) → silence (silence_duration) → repeat
        trimmed to total_duration
    """
    cycle = tone_duration + silence_duration
    repeats = int(total_duration // cycle) + 1  # overshoot, then trim

    cmd = [
        SOX_PATH,
        "-n",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        str(output_path),
        "synth",
        str(tone_duration),
        "sine",
        str(freq_hz),
        "pad",
        "0",
        str(silence_duration),
        "repeat",
        str(repeats),
        "trim",
        "0",
        str(total_duration),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    out_dir = Path("dataset_tonal")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_sox(
        output_path=out_dir / "MARS-20231128T150000Z_1.wav",
        total_duration=60,
        tone_duration=5,
        silence_duration=3,
        freq_hz=2000,
    )

    run_sox(
        output_path=out_dir / "MARS-20231128T150000Z_2.wav",
        total_duration=30,
        tone_duration=2,
        silence_duration=10,
        freq_hz=4000,
    )

    run_sox(
        output_path=out_dir / "MARS-20231128T150000Z_3.wav",
        total_duration=60,
        tone_duration=5,
        silence_duration=3,
        freq_hz=2000,
    )


if __name__ == "__main__":
    main()
