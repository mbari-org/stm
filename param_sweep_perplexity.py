import optuna
import pandas as pd

import topic_model
import labels_to_documents
import cluster
import stft
import conf

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.001, 0.9, log=True)
    beta = trial.suggest_float("beta", 0.001, 0.9, log=True)
    num_topics = trial.suggest_int("num_topics", 2, 20, step=1)
    runs_per_param = 1
    path = conf.model_path

    p_sum = 0
    for i in range(runs_per_param):
        try:
            topic_model.main(alpha=alpha, beta=beta, T=num_topics, W=conf.vocab_size)
            df = pd.read_csv(filepath_or_buffer=path / "perplexity.csv", header=None)
            p = df.sum()[1] / len(df)
            p_sum += p
        except Exception as e:
            print(f"Error during trial {i+1}: {e}")
            continue

    p_avg = p_sum / runs_per_param
    return p_avg

if __name__ == "__main__":
    stft.main()
    cluster.main()
    labels_to_documents.main()

    study = optuna.create_study(direction="minimize", study_name="perplexity_optimization")
    study.optimize(objective, n_trials=3)

    print("Best params:", study.best_params)
    print("Best perplexity:", study.best_value)
    pd.DataFrame([study.best_params]).to_csv(f"./optuna_perplexity_best_params_{conf.window_size}_win_{int(conf.overlap*100)}_ovr_{conf.words_per_doc}_wpd_{conf.g_time}_gtime.csv")