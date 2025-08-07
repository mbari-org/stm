import optuna
import pandas as pd

import topic_model
import visualize
import labels_to_documents
import cluster
import stft
import conf

# Function to calculate perplexity from a CSV
def calculate_perplexity(output_file):
    try:
        df = pd.read_csv(output_file, header=0)
        perplexity = df.iloc[:, 1].sum() / len(df)
        return perplexity
    except Exception as e:
        print(f"[ERROR] Failed to calculate perplexity: {e}")
        return None

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.001, 0.9)
    beta = trial.suggest_float("beta", 0.001, 0.9)
    gamma = trial.suggest_float("gamma", 0.001, 0.9)
    g_time = trial.suggest_int("g_time", 2, 30)

    runs_per_param = 1

    perplexity_sum = 0
    for i in range(runs_per_param):
        try:
            out_dir = conf.model_path / "perplexity_sweeps"/ f"trial_{i+1}_alpha_{alpha}_beta_{beta}_gamma_{gamma}_gtime_{g_time}"
            out_dir.mkdir(parents=True, exist_ok=True)
            trial.set_user_attr("out_dir", out_dir.as_posix())
            topic_model.main(alpha=alpha, beta=beta, gamma=gamma, g_time=g_time, out_dir=out_dir)
            perplexity = calculate_perplexity(out_dir / "perplexity.csv")
            perplexity_sum += perplexity
            visualize.main(alpha=alpha, beta=beta, gamma=gamma, g_time=g_time, model_path=out_dir)
        except Exception as e:
            print(f"Error during trial {i+1}: {e}")
            continue

    perplexity_avg = perplexity_sum / runs_per_param
    return perplexity_avg

if __name__ == "__main__":
    stft.main()
    cluster.main()
    labels_to_documents.main()

    search_space = {
        "alpha": [0.001],
        "beta": [0.001],
        "gamma": [0.001],
        "g_time": [2,3,4,5]
    }
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), study_name="Perplexity_grid")
    study.optimize(objective, n_trials=1, n_jobs=4)
    df = study.trials_dataframe()
    df.to_csv("./optuna_perplexity_grid.csv", index=False)