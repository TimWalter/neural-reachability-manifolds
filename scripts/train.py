import argparse
import json

import wandb
import torch
import optuna
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
from model import Model
from data_sampling.dataset import Dataset
from torcheval.metrics import BinaryConfusionMatrix, R2Score


def log_reachability_data(space, loss, pred, labels: Tensor):
    confusion_mat = BinaryConfusionMatrix(normalize="true").update(pred, (labels != -1).to(torch.int64)).compute().cpu()
    return {f'{space}/Loss ': loss.item(),
            f'{space}/Correct Positives': confusion_mat[1, 1],
            f'{space}/Correct Negatives': confusion_mat[0, 0]}


def log_manipulability_data(space, loss, pred, labels):
    return {f'{space}/Loss ': loss.item(),
            f'{space}/R2Score': R2Score().update(pred, labels).compute().cpu()}


def reachability_loss(pred: Tensor, labels: Tensor, weights: Tensor):
    return torch.nn.BCELoss(weight=weights, reduction='mean')(pred, (labels != -1).to(pred.dtype))


def manipulability_loss(pred: Tensor, labels: Tensor, weights: Tensor):
    mse_loss = torch.nn.MSELoss(reduction='none')(pred, labels)
    weighted_loss = mse_loss * weights
    return weighted_loss.mean()


def infinite_loader(data_loader):
    while True:
        for data in data_loader:
            yield data


model_type_settings = {
    "reachability_classifier": {
        "only_reachable": False,
        "activation_function": torch.nn.Sigmoid(),
        "loss_function": reachability_loss,
        "log_data": log_reachability_data,
    },
    "manipulability_estimator": {
        "only_reachable": True,
        "activation_function": torch.nn.ReLU(),
        "loss_function": manipulability_loss,
        "log_data": log_manipulability_data,
    }
}


def main(model_type: str,
         learning_rate: float,
         batch_size: int,
         epochs: int,
         early_stopping_threshold: int,
         encoder_width: int,
         decoder_width: int,
         encoder_depth: int,
         decoder_depth: int,
         minimum_balance: float,
         trial: optuna.Trial = None
         ):
    settings = model_type_settings[model_type]

    run = wandb.init(project="Capability Maps",
                     config=kwargs if trial is None else {},
                     group=model_type)

    if trial is None:
        folder = Path(__file__).parent.parent / "trained_models" / f"{model_type}_{run.name}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        json.dump(kwargs, open(folder / 'settings.json', 'w'))
    else:
        run.name = f"trial/{trial.number}/{run.name}"

    device = torch.device("cuda")

    training_set = Dataset(Path(__file__).parent.parent / 'data' / 'train', device, batch_size,
                           settings["only_reachable"],
                           minimum_balance)
    validation_set = Dataset(Path(__file__).parent.parent / 'data' / 'val', device, 10000, settings["only_reachable"],
                             0.0)
    validation_iterator = infinite_loader(validation_set)

    model = Model(
        encoder_config={'width': encoder_width, 'depth': encoder_depth},
        decoder_config={'width': decoder_width, 'depth': decoder_depth},
        output_func=settings["activation_function"]

    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = settings["loss_function"]

    min_loss = torch.inf
    early_stopping_counter = 0

    for e in range(epochs):
        for i, (weights, mdhs, poses, labels) in enumerate(tqdm(training_set, desc=f"Training")):
            step = e * len(training_set) + i
            commit = i % 25 != 0

            # weights = weights.to(device, non_blocking=True)
            # mdhs = mdhs.to(device, non_blocking=True)
            # poses = poses.to(device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)

            model.train()
            model.zero_grad()
            pred = model(poses, mdhs)
            loss = loss_function(pred, labels, weights)
            loss.backward()
            optimizer.step()

            run.log(data=settings["log_data"]('train', loss, pred, labels), step=step, commit=commit)

            if not commit:
                with torch.no_grad():
                    model.eval()
                    weights, mdh, poses, labels = next(validation_iterator)
                    # weights = weights.to(device, non_blocking=True)
                    # mdh = mdh.to(device, non_blocking=True)
                    # poses = poses.to(device, non_blocking=True)
                    # labels = labels.to(device, non_blocking=True)
                    pred = model(poses, mdh)
                    loss = loss_function(pred, labels, weights)

                    run.log(data=settings["log_data"]('val', loss, pred, labels), step=step, commit=True)

                if trial is not None:
                    trial.report(loss, e)
                    if trial.should_prune():
                        run.finish()
                        raise optuna.TrialPruned()

                if loss < min_loss:
                    min_loss = loss
                    early_stopping_counter = 0
                    if trial is None:
                        torch.save(model.state_dict(), folder / "reachability_classifier.pth")
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter == early_stopping_threshold:
                        (print('Early Stopping'))
                        run.finish()
                        return min_loss
    run.finish()
    return min_loss


if __name__ == '__main__':
    wandb.login(key="")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='reachability_classifier',
                        choices=['reachability_classifier', 'manipulability_estimator'])
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early_stopping_threshold", type=int, default=-1)
    parser.add_argument("--encoder_width", type=int, default=512, help="hidden size of the LSTM")
    parser.add_argument("--encoder_depth", type=int, default=1, help="# of recurrent LSTM layers")
    parser.add_argument("--decoder_width", type=int, default=128, help="hidden size of the occupancy network")
    parser.add_argument("--decoder_depth", type=int, default=4, help="number of ResNet blocks in the occupancy network")
    parser.add_argument("--minimum_balance", type=float, default=0.5,
                        help="minimum balance of reachable/unreachable samples in each batch")

    args = parser.parse_args()
    kwargs = vars(args)
    print(args)

    main(**kwargs)