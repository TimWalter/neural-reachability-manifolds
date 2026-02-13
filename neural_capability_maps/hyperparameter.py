import argparse
import optuna

from train import main
from neural_capability_maps.model import Model, Torus, OccupancyNetwork, MLP, Shell


def objective(trial):
    print(f"[TRIAL {trial.number}]")
    kwargs.update({
        "hyperparameter": {
            "encoder_config": {
                "dim_encoding": trial.suggest_int("enc_dim_encoding", 128, 1024, step=128),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "drop_prob": trial.suggest_float("drop_prob", 0.0, 1.0)
            },
            "decoder_config": {
                "dim_hidden": trial.suggest_int("dim_hidden", 128, 2*2048, step=128),
                "n_blocks": trial.suggest_int("n_blocks", 1, 12),
            },
        },
        "batch_size": trial.suggest_categorical("batch_size", [1000, 2000, 2500, 4000, 5000, 10000, 25000, 50000, 100000]),
        "lr": trial.suggest_float("lr",  1e-5, 1e-3, log=True),
        "trial": trial,
    })
    return main(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=-1)

    args = parser.parse_args()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage="sqlite:///hyperparameter.sqlite3",
                                study_name=args.model_class)

    args.model_class = eval(args.model_class)

    kwargs = vars(args)
    study.optimize(objective, n_trials=100, n_jobs=1)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
