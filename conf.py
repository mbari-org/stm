# file/directory paths
wav_path = "/Users/bergamaschi/Documents/HumpbackSong/dataset/test/"
out_dir = "./out/"

# spectrogram parameters
window_size = 4096
overlap = 0.9
sample_rate = 32000
power = False
subset = (50, 2000)
sigma = 2
normalize = True
stft_path = out_dir+"stft/"

# clustering parameters
vocab_size = 1000
cluster_type = 'mbk'
cluster_path = out_dir+"cluster/"

# doc parameters
words_per_doc = 30
doc_path = out_dir+"docs/"

# model parameters
target_file = "HBSe_20151207T070326"

num_topics = 5
alpha = 0.01
beta = 0.0001
g_time = 10
cell_space = 0
threads = 4

online = False
online_mint = 5

model_path = out_dir+"model/"
rost_path = "/Users/bergamaschi/rost-cli/bin/"
