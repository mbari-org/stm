import numpy as np
import optuna
import pandas as pd

import topic_model
import labels_to_documents
import cluster
import stft
import conf
from sklearn.metrics import adjusted_rand_score

def get_topic_labels(conf):

    # Get the lookup file to find the target file within the documents
    lookup_file = conf.doc_path / "lookup.csv"
    if not lookup_file.exists():
        print(f"Lookup file {lookup_file} does not exist.")
        return

    # Find the row with the target_file in the lookup file
    target_file_path = conf.doc_path /  f"{conf.target_file}.csv"
    lookup_df = pd.read_csv(lookup_file)
    target_row = lookup_df[lookup_df['filename'] == target_file_path.name]
    if target_row.empty:
        print(f"Target file {target_file_path} not found in lookup file.")
        return

    # Extract start and end timestamps from the lookup
    start_ts = target_row.ms_start.values[0]
    end_ts = target_row.ms_end.values[0]
    theta_model = pd.read_csv(conf.model_path / "theta.csv", header=None).values
    ms_per_doc = int(((conf.window_size / conf.sample_rate) * (1 - conf.overlap) * 1000)) * conf.words_per_doc

    # Extract the relevant portion of theta
    theta = theta_model[start_ts//ms_per_doc:end_ts - ms_per_doc//ms_per_doc]

    # Extract labels
    topics = []
    for i, d in enumerate(theta):
        # Get the max topic for this document
        top_1_topic = np.argmax(d)
        topics.append(top_1_topic)
    return topics

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.001, 0.9, log=True)
    beta = trial.suggest_float("beta", 0.001, 0.9, log=True)
    num_topics = trial.suggest_int("num_topics", 2, 20, step=1)
    runs_per_param = 1

    ari_sum = 0
    for i in range(runs_per_param):
        try:
            topic_model.main(alpha=alpha, beta=beta, T=num_topics, W=conf.vocab_size)
            labels_predicted = get_topic_labels(conf)

            # UNCOMMENT the following line if you have true labels to compare against
            # labels_true = pd.read_csv(conf.wav_path / f"{conf.target_file}.csv", header=None).values.flatten()

            # COMMENT THE FOLLOWING LINE. For now this duplicates predicted labels as true labels, so the ARI will be 1.0
            labels_true = labels_predicted
            ari = adjusted_rand_score(labels_true, labels_predicted)
            ari_sum += ari
        except Exception as e:
            print(f"Error during trial {i+1}: {e}")
            continue

    ari_avg = ari_sum / runs_per_param
    return ari_avg

if __name__ == "__main__":
    stft.main()
    cluster.main()
    labels_to_documents.main()

    study = optuna.create_study(direction="maximize", study_name="ARI_optimization")
    study.optimize(objective, n_trials=1)

    print("Best params:", study.best_params)
    print("Best ARI:", study.best_value)
    pd.DataFrame([study.best_params]).to_csv(f"./optuna_ARI_best_params_{conf.window_size}_win_{int(conf.overlap*100)}_ovr_{conf.words_per_doc}_wpd_{conf.g_time}_gtime.csv")