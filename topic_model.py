from pathlib import Path

import pandas as pd
import numpy as np

import conf
from util import execute
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

    model_path = Path(out_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    top_mod_cmd = ["topics.refine.t",
                   "-i", str(in_dir / f"{target_file}.csv"),
                   "--out.topics=" + str(model_path / "topics.csv"),
                   "--out.topics.ml=" + str(model_path / "topics.maxlikelihood.csv"),
                   "--out.topicmodel=" + str(model_path / "topicmodel.csv"),
                   "--ppx.out=" + str(model_path / "perplexity.csv"),
                   "--logfile=" + str(model_path / "topics.log"),
                   "-V", str(W),
                   "-K", str(T),
                   "--alpha=" + str(alpha),
                   "--beta=" + str(beta),
                   "--threads", str(threads),
                   "--g.time=" + str(g_time),
                   "--cell.space=" + str(cell_space)]

    if online:
        top_mod_cmd.extend([
            "words.bincount",
            "--online",
            "--out.topics.online=" + str(model_path / "topics.online.csv"),
            "--out.ppx.online=" + str(model_path / "perplexity.online.csv"),
            "--online.mint", str(online_mint)
        ])

    bin_cnt_cmd = ["-i", str(model_path / "topics.maxlikelihood.csv"),
                   "-o", str(model_path / "topics.hist.csv"),
                   "-V", str(T)]

    if conf.use_docker:
        execute(top_mod_cmd, volume_mount=conf.model_path.parent, use_docker=True)
    else:
        execute(rost_path + top_mod_cmd)
        execute(rost_path + bin_cnt_cmd)

    topic_model = pd.read_csv(model_path / "topicmodel.csv", header=None).values
    topic_hist = pd.read_csv(model_path / "topics.csv", header=None).drop(0, axis=1).values

    phi = np.zeros((T, W))
    for z in range(T):
        denominator = np.sum(topic_model[z]) + (W * beta)
        for w in range(W):
            numerator = topic_model[z][w] + beta
            phi[z][w] = numerator / denominator
    write_csv(phi, model_path / "phi.csv")

    D = len(topic_hist)
    theta = np.zeros((D, T))
    for d in range(D):
        denominator = np.sum(topic_hist[d]) + (T * alpha)
        for z in range(T):
            numerator = topic_hist[d][z] + alpha
            theta[d][z] = numerator / denominator

    write_csv(theta, model_path / "theta.csv")

    df = pd.read_csv(model_path / "perplexity.csv", header=None)
    print(f"Avg Per Doc Perplexity = {df.sum()[1] / len(df)}")


if __name__ == "__main__":
    main()