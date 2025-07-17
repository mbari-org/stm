import visualize
import conf
from pathlib import Path

for sound_file in Path(conf.wav_path).glob("*.wav"):
    print(f"Processing {sound_file.name}")
    visualize.main(target_file=str(sound_file.stem),
                   times=conf.times,
                   model_path=conf.model_path,
                   stft_path=conf.stft_path,
                   doc_path=conf.doc_path,
                   window_size=conf.window_size,
                   overlap=conf.overlap,
                   fs=conf.sample_rate,
                   subset=conf.subset,
                   words_per_doc=conf.words_per_doc)
