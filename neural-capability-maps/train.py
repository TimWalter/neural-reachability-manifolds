import argparse
import json

import wandb
import torch
import optuna
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
from model import ReachabilityClassifier as Model
from dataset import Dataset, SingleDataset


def log_data(space, loss, pred, labels: Tensor):
    predicted_labels = pred > 0.5
    labels = labels.bool()
    true_positives = (predicted_labels & labels).sum().item() / labels.shape[0]
    true_negatives = (~predicted_labels & ~labels).sum().item() / labels.shape[0]
    false_positives = (predicted_labels & ~labels).sum().item() / labels.shape[0]
    false_negatives = (~predicted_labels & labels).sum().item() / labels.shape[0]

    return {f'{space}/Loss ': loss.item(),
            f'{space}/True Positives': true_positives,
            f'{space}/True Negatives': true_negatives,
            f'{space}/False Positives': false_positives,
            f'{space}/False Negatives': false_negatives
            }


def main(epochs: int,
         batch_size: int,
         early_stopping: int,
         model_hyperparameter: dict = None,
         optimiser_hyperparameter: dict = None,
         trial: optuna.Trial = None
         ):
    if model_hyperparameter is None:
        model_hyperparameter = {
            'encoder_config': {'width': 512, 'depth': 1},
            'decoder_config': {'width': 128, 'depth': 4}
        }
        #model_hyperparameter = {
        #    'encoder_config': {"nhead": 8, "dim_feedforward": 640, "dropout": 0.1},
        #    'decoder_config': {"n_heads": 8, "ff_dim": 640, "dropout": 0.1},
        #    "latent_morph": 160, "latent_pose": 160,
        #    "num_encoder_blocks": 4, "num_decoder_blocks": 4
        #}
    if optimiser_hyperparameter is None:
        optimiser_hyperparameter = {'lr': 5e-4}
    settings = {'epochs': epochs, 'batch_size': batch_size, 'early_stopping': early_stopping}
    settings['model_hyperparameter'] = model_hyperparameter
    settings['optimiser_hyperparameter'] = optimiser_hyperparameter

    device = torch.device("cuda")
    run = wandb.init(project="Capability Maps",
                     config=settings if trial is None else {},
                     group="reachability_classifier")

    if trial is None:
        folder = Path(__file__).parent.parent / "trained_models" / f"reachability_classifier_{run.name}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        json.dump(settings, open(folder / 'settings.json', 'w'))
    else:
        run.name = f"trial/{trial.number}/{run.name}"

    training_set = Dataset(Path(__file__).parent.parent / 'data' / 'train', batch_size, True, True)
    validation_set = Dataset(Path(__file__).parent.parent / 'data' / 'val', batch_size, False, False)

    model = Model(**model_hyperparameter).to(device)
    optimizer = optim.Adam(model.parameters(), **optimiser_hyperparameter)

    loss_function = torch.nn.BCELoss(reduction='mean')

    min_loss = torch.inf
    early_stopping_counter = 0
    for e in range(epochs):
        for batch_idx, (morph, pose, label) in enumerate(tqdm(training_set, desc=f"Training")):
            step = e * len(training_set) + batch_idx

            morph = morph.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            model.train()
            model.zero_grad()
            pred = model(pose, morph)
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()

            run.log(data=log_data('train', loss, pred, label), step=step, commit=False)

            model.eval()
            val_morph, val_pose, val_label = validation_set.get_random_batch()
            val_morph = val_morph.to(device, non_blocking=True)
            val_pose = val_pose.to(device, non_blocking=True)
            val_label = val_label.to(device, non_blocking=True)
            with torch.no_grad():
                validation_pred = model(val_pose, val_morph)
            validation_loss = loss_function(validation_pred, val_label)

            run.log(data=log_data('val', validation_loss, validation_pred, val_label), step=step, commit=True)

            if trial is not None:
                trial.report(validation_loss, step)
                if trial.should_prune():
                    run.finish()
                    raise optuna.TrialPruned()

            if validation_loss < min_loss:
                min_loss = validation_loss
                early_stopping_counter = 0
                if trial is None:
                    torch.save(model.state_dict(), folder / "reachability_classifier.pth")
            else:
                early_stopping_counter += 1
                if early_stopping_counter == early_stopping:
                    (print('Early Stopping'))
                    run.finish()
                    return min_loss
    run.finish()
    return min_loss


if __name__ == '__main__':
    wandb.login(key="")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=-1)

    args = parser.parse_args()
    kwargs = vars(args)
    print(args)

    main(**kwargs)
