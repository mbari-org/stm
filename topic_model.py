import pandas as pd
import numpy as np
import subprocess


def execute(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def flip_list(l):
    temp = []
    for i in range(len(l[0])):
        temp.append(l[:,i])
    return np.array(temp)


def main():
    documents_path = "/Users/bergamaschi/Documents/PAM/documents_HBSe_20151207T070326.csv"
    model_path = "/Users/bergamaschi/Documents/PAM/ROST_output/"

    W = 10000  # vocabulary size
    T = 10  # topic size
    alpha = 0.1  # sparsity of theta
    beta = 1  # sparsity of phi
    online_mint = 5  # min time in ms to spend between new observation timestep
    threads = 4
    g_time = 0  # depth of temporal neighborhood in #cells
    g_space = 0  # depth of spacial neighborhood in #cells
    cell_space = 0  # cell width in time dim

    gamma = 0.0000001  # ???
    grow_topics_size = 1  # ???

    top_mod_cmd = ["/Users/bergamaschi/rost-cli/bin/topics.refine.t",
                   "-i", documents_path,
                   "--out.topics=" + model_path + "topics.csv",
                   "--out.topics.ml=" + model_path + "topics.maxlikelihood.csv",
                   "--out.topicmodel=" + model_path + "topicmodel.csv",
                   "--ppx.out=" + model_path + "perplexity.csv",
                   "--logfile=" + model_path + "topics.log",
                   "--out.topics.online=" + model_path + "topics.online.csv",
                   "--out.ppx.online=" + model_path + "perplexity.online.csv",
                   "-V", str(W),
                   "-K", str(T),
                   "--alpha=" + str(alpha),
                   "--beta=" + str(beta),
                   "--online",
                   "--online.mint", str(online_mint),
                   "--threads", str(threads),
                   "--g.time=" + str(g_time),
                   "--g.space=" + str(g_space),
                   "--cell.space=" + str(cell_space)]

    bin_cnt_cmd = ["/Users/bergamaschi/rost-cli/bin/words.bincount",
                   "-i", model_path + "topics.maxlikelihood.csv",
                   "-o", model_path + "topics.hist.csv",
                   "-V", str(T)]

    print(top_mod_cmd)
    print(bin_cnt_cmd)

    execute(top_mod_cmd)
    execute(bin_cnt_cmd)

    topic_model = pd.read_csv(model_path + "topicmodel.csv", header=None).values
    topic_hist = pd.read_csv(model_path + "topics.hist.csv", header=None).drop(0, axis=1).values

    # ===>compute phi<===
    phi = np.empty((T, W))

    for z in range(T):
        denominator = np.sum(topic_model[z]) + (W * beta)
        for w in range(W):
            numerator = topic_model[z][w] + beta
            phi[z][w] = numerator / denominator

    # ===>compute theta<===
    D = len(topic_hist)  # number of documents
    theta = np.empty((D, T))

    for d in range(D):
        denominator = np.sum(topic_hist[d]) + (T * alpha)
        for z in range(T):
            numerator = topic_hist[d][z] + alpha
            theta[d][z] = numerator / denominator


if __name__ == "__main__":
    main()

