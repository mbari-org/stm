# 🎵 STM: Sound Topic Modeling

Tools for discovering units, phrases, and themes in audio files using Perch2 embeddings and/or PCEN spectrogram features,
with a topic model.  Outputs can be edited using the popular [Raven](https://www.ravensoundsoftware.com/) then trained
for classification using a linear probe.  

The topic modeling tool used in this code is Realtime Online Spatiotemporal Topic Modeling [rost-cli tool](https://gitlab.com/warplab/rost-cli).

Local runs use the Docker image by default. To use prebuilt binaries instead, host a gzip tarball whose root contains `bin/topics.refine.t` and `bin/words.bincount`, then point `ensure_rost_cli()` at it (`STM_ROST_CLI_URL` or the `url` argument). It caches the `bin/` directory under `~/.cache/stm/rost-cli` and returns that path for `TopicModelRunner(use_docker=False, rost_path=...)`. On Debian and Colab, install the shared libraries those binaries link against before running them: `libboost-all-dev`, `libflann-dev`, `libfftw3-dev`, `libopencv-dev`, `libsndfile1-dev`, `libgstreamer-plugins-base1.0-dev`, `libgstreamer1.0-0`, and `libhdf5-dev`.

Build the archive from a compiled tree (`tar -C /app/rost-cli -czf rost-cli-linux-x86_64.tar.gz bin`) and upload it with:

```bash
pip install 'stm[deploy]'
python -m stm.topicmodel.rost_deploy s3://BUCKET/prefix rost-cli-linux-x86_64.tar.gz
```

The command prints the HTTPS object URL. Set that URL as `STM_ROST_CLI_URL`. The bucket or object must be publicly readable for `ensure_rost_cli()` to download it.

There are many parameters that can be adjusted to tune the model to the data.  Default parameters are set to work well 
with underwater sounds, but can be adjusted for other applications.  All parameters are in the `conf.yaml` file.


Start with these example Google Colab notebooks. You will need a valid google email to use these. 
All of these example use a provided short audio recording with a humpback song.  

TODO: add description of Perch2 and PCEN features.

### Step 1. Discover units, phrases, and themes in audio files 
 
* [Train with Perch2](https://colab.research.google.com/github/mbari-org/stm/blob/main/stm/notebooks/train_topic_model_perch2.ipynb) 
  — Computes embeddings in the audio with the Google Perch2 model then train a topic model on those embeddings to discover themes and phrases.
* [Train with PCEN](https://colab.research.google.com/github/mbari-org/stm/blob/main/stm/notebooks/train_topic_model_pcen.ipynb) 
  — Compute PCEN spectrogram features and train a topic model without Perch2. This is faster than Perch2 and finer-grained. Best for unit-level classification.
* [Train with Perch2 and PCEN](https://colab.research.google.com/github/mbari-org/stm/blob/main/stm/notebooks/train_topic_model.ipynb) 
  — Computes embeddings in the audio with the Google Perch2 model and PCEN features, then train a topic model.

### Step 2. Edit topic model output in Raven Lite or Raven Pro


### Step 3.  

* [Extract Raven units](https://colab.research.google.com/github/mbari-org/stm/blob/main/stm/notebooks/extract_raven_units.ipynb) 
  — Parse Raven selection tables and export labeled unit clips.
* [Train a linear probe on Raven units](https://colab.research.google.com/github/mbari-org/stm/blob/main/stm/notebooks/train_raven_units.ipynb) 
  — embed exported units with Perch2 and train a linear probe.


# Acknowledgements

Foundation for the application of topic modeling was the excellent work done by a former MBARI intern project 
[hb-song-analysis Thomas Bergamaschi](https://github.com/tbergama/hb-song-analysis)

