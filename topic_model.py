from pathlib import Path

import pandas as pd
import numpy as np

import conf
from util import execute
from util import write_csv


def main(in_dir=conf.doc_path, out_dir=conf.model_path,
         rost_path=conf.rost_path, target_file=conf.combined_document_file,
         W=conf.vocab_size, T=conf.num_topics, alpha=conf.alpha, beta=conf.beta,
         g_time=conf.g_time, cell_space=conf.cell_space, threads=conf.threads,
         online=conf.online, online_mint=conf.online_mint, gamma=conf.gamma):

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
    :param gamma: used if T is None, to control the growth of topics
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
                   "--alpha=" + str(alpha),
                   "--beta=" + str(beta),
                   "--threads", str(threads),
                   "--g.time=" + str(g_time),
                   "--cell.space=" + str(cell_space)]

    if T is not None:
        top_mod_cmd.extend(["-K", str(T)])
    else:
        top_mod_cmd.extend(["--grow.topics.size=true",f"--gamma={gamma}",])

    print(" ".join(top_mod_cmd))

    if online:
        top_mod_cmd.extend([
            "--online",
            "--out.topics.online=" + str(model_path / "topics.online.csv"),
            "--out.ppx.online=" + str(model_path / "perplexity.online.csv"),
            "--online.mint", str(online_mint)
        ])

    if conf.use_docker:
        execute(top_mod_cmd, volume_mount=conf.model_path.parent, use_docker=True)
    else:
        execute(rost_path + top_mod_cmd)

    if T is None:
        top_df = pd.read_csv(model_path / "topics.csv", header=None)
        top_df.index = top_df[0]
        # Find the maximum topic values in the entire dataframe
        T = int(top_df.drop(0, axis=1).max().max() + 1)

    bin_cnt_cmd = ["words.bincount",
                   "-i", str(model_path / "topics.maxlikelihood.csv"),
                   "-o", str(model_path / "topics.hist.csv"),
                   "-V", str(T)]

    if conf.use_docker:
        execute(bin_cnt_cmd, volume_mount=conf.model_path.parent, use_docker=True)
    else:
        execute(rost_path + bin_cnt_cmd)

    topic_model = pd.read_csv(model_path / "topicmodel.csv", header=None).values
    topic_hist = pd.read_csv(model_path / "topics.hist.csv", header=None).drop(0, axis=1).values

    phi = np.zeros((T, W))
    for z in range(T):
        denominator = np.sum(topic_model[z]) + (W * beta)
        for w in range(W):
            numerator = topic_model[z][w] + beta
            if denominator == 0:
                phi[z][w] = 0.0
            else:
                phi[z][w] = numerator / denominator
    write_csv(phi, model_path / "phi.csv")

    D = len(topic_hist)
    theta = np.zeros((D, T))
    for d in range(D):
        denominator = np.sum(topic_hist[d]) + (T * alpha)
        for z in range(T):
            numerator = topic_hist[d][z] + alpha
            if denominator == 0:
                theta[d][z] = 0.0
            else:
                theta[d][z] = numerator / denominator
    write_csv(theta, model_path / "theta.csv")

    # Open theta and clamp any values less than .7 to 0 and those that are greater than .7 to 1
    # theta_df = pd.read_csv(model_path / "theta.csv", header=None)
    # for col in theta_df.columns:
    #     theta_df[col] = theta_df[col].apply(lambda x: 1.0 if x >= 0.6 else 0.0)
    # pd.DataFrame(theta_df).to_csv(model_path / "theta.csv", index=False, header=False)

    df = pd.read_csv(model_path / "perplexity.csv", header=None)
    print(f"Avg Per Doc Perplexity = {df.sum()[1] / len(df)}")


if __name__ == "__main__":
    main()