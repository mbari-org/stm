# file/directory paths
wav_path = "/Users/bergamaschi/Documents/HumpbackSong/dataset/test/"
out_dir = "./out/"

# spectrogram parameters
window_size = 1024
overlap = 0.5
subset = (50, 2000)
sample_rate = 32000
stft_path = out_dir+"stft/"

# clustering parameters
vocab_size = 1000
cluster_type = 'mbk'
cluster_path = out_dir+"cluster/"

# doc parameters
words_per_doc = 32
doc_path = out_dir+"docs/"

# model parameters
target_file = "HBSe_20151207T070326"

num_topics = 5
alpha = 0.1
beta = 0.001
g_time = 2
cell_space = 0
threads = 4

online = False
online_mint = 5

model_path = out_dir+"model/"
rost_path = "/Users/bergamaschi/rost-cli/bin/"
