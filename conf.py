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
window_size = 2048 # Window size for STFT
overlap = 0.9
sample_rate = 32000
subset = (50, 4000) # Subset frequency range for spectrogram
sigma = None
normalize = False
use_pcen = True  # Use PCEN for normalization https://librosa.org/doc/0.11.0/generated/librosa.pcen.html
pcen_gain = 0.25  # Gain for PCEN, generally between 0.1 and 0.5
pcen_bias = 2  # Bias for PCEN
pcen_time_constant = 0.4  # Time constant for PCEN, in seconds

# clustering parameters
vocab_size = 100  # Vocabulary size for clustering
whiten = None  # 'pca', 'std', or None
cluster_type = 'mbk' # Clustering type: 'mbk', or 'kmeans

# doc parameters
words_per_doc = 32

# model parameters
num_topics = None # If None, the model will grow topics dynamically.
alpha = 0.001 # Lower = each document is more likely to be dominated by a small number of topics (i.e., sparse topic distribution).
beta = 0.01 # Lower = each topic is more likely to be dominated by a small number of words (i.e., sparse word distribution).
gamma = 0.001  # Used if num_topics is None, to control the growth of topics.
g_time = 2  # Depth of temporal neighborhood in cells
cell_space = 0 # cell width in space dim
threads = 64

online = False
online_mint = 5

# visualization parameters
# To see the whole spectrogram, set times to None
# times = None
times = (16, 77) # Time range for visualization in seconds

# Uncomment the following lines to visualize a specific time range
# import datetime
# time_start = "00:02:30"
# time_end = "00:03:30"
# time_start_sec = datetime.datetime.strptime(time_start, "%H:%M:%S")
# time_end_sec = datetime.datetime.strptime(time_end, "%H:%M:%S")
# times = (time_start_sec.hour * 3600 + time_start_sec.minute * 60 + time_start_sec.second,
#         time_end_sec.hour * 3600 + time_end_sec.minute * 60 + time_end_sec.second)