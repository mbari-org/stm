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
combined_document_file = "all_docs"
target_file = "HBSe_20151207T070326"  # Single target file for testing
model_path = out_dir / "model"

# spectrogram parameters
window_size = 2048
overlap = 0.95
sample_rate = 32000
power = False
subset = (50, 2000)
sigma = 2
normalize = False
use_pcen = True  # Use PCEN for normalization
pcen_gain = 0.5  # Gain for PCEN, if using PCEN
pcen_bias = 2  # Bias for PCEN, if using PCEN

# clustering parameters
vocab_size = 100
whiten = None  # 'pca', 'std', or None
cluster_type = 'mbk'

# doc parameters
words_per_doc = 32

# model parameters
num_topics = None # If None, the model will grow topics dynamically.
alpha = 0.01 # Lower = each document is more likely to be dominated by a small number of topics (i.e., sparse topic distribution).
beta = 0.1 # Lower = each topic is more likely to be dominated by a small number of words (i.e., sparse word distribution).
g_time = 2  # Depth of temporal neighborhood in cells
cell_space = 0 # cell width in space dim
threads = 16

online = False
online_mint = 5

# visualization parameters
times = (16, 77)
