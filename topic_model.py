import pandas as pd
import numpy as np
import subprocess
import os
import csv


def execute(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def flip_list(l):
    temp = []
    for i in range(len(l[0])):
        temp.append(l[:,i])
    return np.array(temp)


def ensure_dir(fname):
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)


def write_csv(data, fname):
    ensure_dir(fname)
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


def run(in_dir, filename, W, T, alpha, beta,
        g_time, cell_space, threads=4,
        online=False, online_mint=5):

    # TODO: Add chinese restaurant problem option

    """

    :param in_dir: directory containing document CSV
    :param filename: name of csv
    :param W: vocabulary size
    :param T: topic size
    :param alpha: sparsity of theta
    :param beta: sparsity of phi
    :param threads: number of threads
    :param g_time: depth of temporal neighborhood
    :param cell_space: cell width in time dimension
    :param online: run ROST online
    :param online_mint: min time (in ms) to spend between observation time steps
    :return: path to directory containing topic model CSVs
    """

    model_path = "./out/model/"

    top_mod_cmd = ["/Users/bergamaschi/rost-cli/bin/topics.refine.t",
                   "-i", in_dir+filename+".csv",
                   "--out.topics=" + model_path + "topics.csv",
                   "--out.topics.ml=" + model_path + "topics.maxlikelihood.csv",
                   "--out.topicmodel=" + model_path + "topicmodel.csv",
                   "--ppx.out=" + model_path + "perplexity.csv",
                   "--logfile=" + model_path + "topics.log",
                   "-V", str(W),
                   "-K", str(T),
                   "--alpha=" + str(alpha),
                   "--beta=" + str(beta),
                   "--threads", str(threads),
                   "--g.time=" + str(g_time),
                   "--cell.space=" + str(cell_space)]

    if online:
        top_mod_cmd.extend(["--online",
                            "--out.topics.online=" + model_path + "topics.online.csv",
                            "--out.ppx.online=" + model_path + "perplexity.online.csv",
                            "--online.mint", str(online_mint)])

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
    write_csv(phi, model_path+"phi.csv")

    # ===>compute theta<===
    D = len(topic_hist)  # number of documents
    theta = np.empty((D, T))

    for d in range(D):
        denominator = np.sum(topic_hist[d]) + (T * alpha)
        for z in range(T):
            numerator = topic_hist[d][z] + alpha
            theta[d][z] = numerator / denominator

    write_csv(theta, model_path+"theta.csv")




