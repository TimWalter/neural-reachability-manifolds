import argparse
import optuna

from train import main


def objective(trial):
    print(f"[TRIAL {trial.number}]")
    kwargs.update({
        "optimiser_hyperparameter": {"lr": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True), },
        "model_hyperparameter": {
            "encoder_config": {
                "width": trial.suggest_int("encoder_width", 256, 1024, step=256),
                "depth": trial.suggest_int("encoder_depth", 1, 4),
            },
            "decoder_config": {
                "width": trial.suggest_int("decoder_width", 64, 256, step=32),
                "depth": trial.suggest_int("decoder_depth", 4, 5),
            },
        },
        "trial": trial,
    })
    return main(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=5)
    args = parser.parse_args()
    kwargs = vars(args)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage="sqlite:///hyperparameter.sqlite3",
                                study_name="reachability_classifier")

    study.optimize(objective, n_trials=1, n_jobs=1)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
