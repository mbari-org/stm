#!/usr/bin/env bash

python stft.py
python cluster.py
python labels_to_documents.py
python topic_model.py
python visualize.py
