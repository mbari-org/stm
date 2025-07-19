import librosa

import visualize
import conf
from pathlib import Path

# Open each wav file in the model_path and visualize it in chunks of 60 seconds
for sound_file in Path(conf.wav_path).glob("*.wav"):
    y, sr = librosa.load(sound_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duration: {duration} seconds")
    times = [(i, min(i + 60, duration)) for i in range(0, int(duration), 60)]
    for start, end in times:
        print(f"Processing chunk from {start} to {end} seconds")
        visualize.main(target_file=str(sound_file.stem),
                       times=[start, end],
                       model_path=conf.model_path,
                       stft_path=conf.stft_path,
                       doc_path=conf.doc_path,
                       window_size=conf.window_size,
                       overlap=conf.overlap,
                       fs=conf.sample_rate,
                       subset=conf.subset,
                       words_per_doc=conf.words_per_doc)
