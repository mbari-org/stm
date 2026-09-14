# 🎵 STM2: Sound Topic Modeling

Tools for discovering units, phrases, and themes in audio files using Perch2 embeddings and/or PCEN spectrogram features,
with a topic model.  Outputs can be edited using the popular [Raven](https://www.ravensoundsoftware.com/) then trained
for classification using a linear probe.  

The topic modeling tool used in this code is Realtime Online Spatiotemporal Topic Modeling [rost-cli tool](https://gitlab.com/warplab/rost-cli).

There are many parameters that can be adjusted to tune the model to the data.  Default parameters are set to work well 
with underwater sounds, but can be adjusted for other applications.  All parameters are in the `conf.yaml` file.


Start with these example Google Colab notebooks. You will need a valid google email to use these. 
All of these example use a provided short audio recording with a humpback song.  

TODO: add description of Perch2 and PCEN features.

### Step 1. Discover units, phrases, and themes in audio files 
 
* [Train with Perch2](perchtopic/notebooks/train_topic_model_perch2.ipynb) 
  — Computes embeddings in the audio with the Google Perch2 model then train a topic model on those embeddings to discover themes and phrases.
* [Train with PCEN](perchtopic/notebooks/train_topic_model_pcen.ipynb) 
  — Compute PCEN spectrogram features and train a topic model without Perch2. This is faster than Perch2 and finer-grained. Best for unit-level classification.
* [Train with Perch2 and PCEN](perchtopic/notebooks/train_topic_model.ipynb) 
  — Computes embeddings in the audio with the Google Perch2 model and PCEN features, then train a topic model.

### Step 2. Edit topic model output in Raven Lite or Raven Pro


### Step 3.  

* [Extract Raven units](perchtopic/notebooks/extract_raven_units.ipynb) 
  — Parse Raven selection tables and export labeled unit clips.
* [Train a linear probe on Raven units](perchtopic/notebooks/train_raven_units.ipynb) 
  — embed exported units with Perch2 and train a linear probe.


# Acknowledgements

Foundation for the application of topic modeling was the excellent work done by a former MBARI intern project 
[hb-song-analysis Thomas Bergamaschi](https://github.com/tbergama/hb-song-analysis)

