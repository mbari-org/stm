from pathlib import Path

# file/directory paths
wav_path = Path.cwd() / "dataset"
rost_path = "./rost-cli/bin/"
use_docker = True # Set to False if using local installation with rost_path
rost_docker_image = "rost-cli:latest"
out_dir = Path.cwd() / "output"
stft_path = out_dir / "stft/"
cluster_path = out_dir / "cluster/"
doc_path = out_dir / "docs"
target_file = "HBSe_20151207T070326"
model_path = out_dir / "model"

# spectrogram parameters
window_size = 1024
overlap = 0.5
sample_rate = 32000
power = True
subset = (50, 2000)
sigma = 2
normalize = True

# clustering parameters
vocab_size = 1000
whiten = None  # 'pca', 'std', or None
cluster_type = 'mbk'

# doc parameters
words_per_doc = 32

# model parameters
num_topics = 10
alpha = 0.01
beta = 0.1
g_time = 2
cell_space = 0
threads = 4

online = False
online_mint = 5

# visualization parameters
times = (16, 77)
