import visualize
import conf
from pathlib import Path

for sound_file in Path(conf.wav_path).glob("*.wav"):
    visualize.main(target_file=str(sound_file.stem))