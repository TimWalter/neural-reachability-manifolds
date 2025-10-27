import argparse
import optuna

from scripts.train import main

def objective(trial):
    print(f"[TRIAL{trial.number}]")
    kwargs.update({
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "encoder_width": trial.suggest_int("encoder_width", 256, 1024, step=256),
        "decoder_width": trial.suggest_int("decoder_width", 64, 256, step=32),
        "encoder_depth": trial.suggest_int("encoder_depth", 1, 4),
        "decoder_depth": trial.suggest_int("decoder_depth", 4, 5),
        "minimum_balance": trial.suggest_float("minimum_balance", 0.001, 0.5, log=True),
        "trial": trial,
    })
    return main(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="reachability_classifier", choices=["reachability_classifier", "manipulability_estimator"])
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--early_stopping_threshold", type=int, default=5)
    args = parser.parse_args()
    kwargs = vars(args)


    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage="sqlite:///nf_capa.sqlite3",
                                study_name=args.model_type)

    study.optimize(objective, n_trials=50, n_jobs=1)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
