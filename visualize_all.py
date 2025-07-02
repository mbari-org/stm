import visualize
import conf
from pathlib import Path

for sound_file in Path(conf.wav_path).glob("*.wav"):
    print(f"Processing {sound_file.name}")
    visualize.main(target_file=str(sound_file.stem))