import argparse
import optuna

from train import main
from neural_capability_maps.model import Model, Torus, OccupancyNetwork, MLP, Shell


def objective(trial):
    print(f"[TRIAL {trial.number}]")
    kwargs.update({
        "hyperparameter": {
            "encoder_config": {
                "dim_encoding": 128,#trial.suggest_int("enc_dim_encoding", 128, 1024, step=128),
                "num_layers": 1,#trial.suggest_int("num_layers", 1, 4),
                "drop_prob": 0.0#trial.suggest_float("drop_prob", 0.0, 1.0)
            },
            "decoder_config": {
                "dim_hidden": trial.suggest_int("dim_hidden", 128, 2*2048, step=128),
                "n_blocks": trial.suggest_int("n_blocks", 1, 12),
            },
            "fourier_config": {
                "dim_encoding": trial.suggest_int("fourier_dim_encoding", 64, 256, step=8),
                "std": trial.suggest_float("std", 0.1, 5.0)
            }
        },
        "trial": trial,
    })
    return main(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage="sqlite:///hyperparameter.sqlite3",
                                study_name=args.model_class)

    args.model_class = eval(args.model_class)

    kwargs = vars(args)
    study.optimize(objective, n_trials=20, n_jobs=1)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
