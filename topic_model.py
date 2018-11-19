import pandas as pd
import numpy as np

import conf
from util import execute
from util import ensure_dir
from util import write_csv


def main(in_dir=conf.doc_path, out_dir=conf.model_path,
         rost_path=conf.rost_path, target_file=conf.target_file,
         W=conf.vocab_size, T=conf.num_topics, alpha=conf.alpha, beta=conf.beta,
         g_time=conf.g_time, cell_space=conf.cell_space, threads=conf.threads,
         online=conf.online, online_mint=conf.online_mint):

    # TODO: Add chinese restaurant process option

    """

    :param in_dir: directory containing document CSV
    :param rost_path: path to ROST executables
    :param model_path: path to directory where model output should be saved
    :param target_file: name of documents csv file
    :param W: vocabulary size
    :param T: number of topics
    :param alpha: sparsity of theta
    :param beta: sparsity of phi
    :param threads: number of threads
    :param g_time: depth of temporal neighborhood
    :param cell_space: cell width in time dimension
    :param online: run ROST online
    :param online_mint: min time (in ms) to spend between observation time steps
    """

    model_path = out_dir
    ensure_dir(model_path)

    top_mod_cmd = [rost_path +"topics.refine.t",
                   "-i", in_dir + target_file + ".csv",
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

    bin_cnt_cmd = [rost_path +"words.bincount",
                   "-i", model_path + "topics.maxlikelihood.csv",
                   "-o", model_path + "topics.hist.csv",
                   "-V", str(T)]

    print(top_mod_cmd)
    print(bin_cnt_cmd)

    execute(top_mod_cmd)
    execute(bin_cnt_cmd)

    topic_model = pd.read_csv(model_path + "topicmodel.csv", header=None).values
    topic_hist = pd.read_csv(model_path + "topics.hist.csv", header=None).drop(0, axis=1).values

    # TODO: make compute_phi function
    # ===>compute phi<===
    phi = np.zeros((T, W))

    for z in range(T):
        denominator = np.sum(topic_model[z]) + (W * beta)
        for w in range(W):
            numerator = topic_model[z][w] + beta
            phi[z][w] = numerator / denominator
    write_csv(phi, model_path + "phi.csv")

    # TODO: make compute_theta function
    # ===>compute theta<===
    D = len(topic_hist)  # number of documents
    theta = np.zeros((D, T))

    for d in range(D):
        denominator = np.sum(topic_hist[d]) + (T * alpha)
        for z in range(T):
            numerator = topic_hist[d][z] + alpha
            theta[d][z] = numerator / denominator

    write_csv(theta, model_path + "theta.csv")

    df = pd.read_csv(filepath_or_buffer=model_path + "perplexity.csv",
                     header=None)
    print("Avg Per Doc Perplexity = {}".format( (df.sum()[1])/len(df) ))


if __name__ == "__main__":
    main()

