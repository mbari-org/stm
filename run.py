import stft
import cluster
import labels_to_documents
import topic_model
import config as cnf

# TODO: change file types from pickle to JSON


def preprocessing():

    in_dir = cnf.WAVPATH

    # stft parameters to iterate over
    window_size = [1024]
    overlap = [0.5]
    subset = (50, 2000)

    # kmeans parameters
    k_vals = [10]

    for win in window_size:
        for ovr in overlap:
            stft_dir = stft.run(in_dir, window_size=win, overlap=ovr, subset=subset)
            for k in k_vals:
                cluster.run(stft_dir, k)


def doc_building():

    # ===>Build Documents<===

    # document parameters
    words_per_doc = 32
    window_size = 1024
    overlap = 0.5
    subset = (50, 2000)
    k_val = 10000
    labels_dir = "./out/stft/win_{}/ovr_{}/sub_{}/k_{}/labels/".format(window_size, overlap, subset, k_val)

    docs_dir = labels_to_documents.run(in_dir=labels_dir, words_per_doc=words_per_doc,
                                       window_size=window_size, overlap=overlap)

    print(docs_dir)


def modeling():

    # ===>Run ROST<===

    # model parameters
    filename = 'HBSe_20151022T015622'
    words_per_doc = 32
    in_dir = './out/model/{}_word_docs/'.format(words_per_doc)

    vocab_size = 10000
    num_topics = 10
    alpha = 0.1
    beta = 0.1
    g_time = 0
    cell_space = 0


    # online parameters
    online = False
    online_mint = 5

    topics_dir = topic_model.run(in_dir=in_dir, filename=filename, W=vocab_size, T=num_topics,
                                 alpha=alpha, beta=beta,
                                 g_time=g_time, cell_space=cell_space,
                                 online=online, online_mint=online_mint)

def classification():

    # ===>Preprocessing<===

    # For entire data set:
    #   Compute stft
    #   Subset
    #   Gaussian filter

    # For training set:
    #   Compute avg, std
    #   Normalize
    #   Cluster and save codebook/labels
    #   Group labels into documents
    #   Label documents


    # For testing set:
    #   Normalize using training avg/std
    #   Quantize using training codebook
    #   Group labels into documents
    #   Label documents

    # ===>Training<===

    # Run ROST on training documents
    # Compute marginal distribution: P(class | k = z)
    # Plot marginal distribution
    # Compute conditional distribution: P(k = z | class)
    # Assign topics their highest probability label

    # ===>Testing<===

    # Run ROST on testing docs using:
    #   --in.topicmodel <in.topicmodel.csv>
    #   --topicmodel.update=0

    # Compute P(z=k | d)

if __name__ == '__main__':
    # preprocessing()
    doc_building()
    # modeling()




