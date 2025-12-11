import argparse
import json

import wandb
import torch
import optuna
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from neural_capability_maps.model import ReachabilityClassifier, SingleMorphologySIREN, MLP, TransTest
from neural_capability_maps.dataset import Dataset, SingleDataset
from data_sampling.representations import continuous_to_rotation_matrix, rotation_matrix_to_rotation_vector



def calculate_metrics(pred, labels):
    predicted_labels = torch.nn.Sigmoid()(pred) > 0.5
    labels = labels.bool()

    true_positives = (predicted_labels & labels).sum().item() / labels.sum() * 100
    true_negatives = (~predicted_labels & ~labels).sum().item() / (~labels).sum() * 100
    false_positives = (predicted_labels & ~labels).sum().item() / (~labels).sum() * 100
    false_negatives = (~predicted_labels & labels).sum().item() / labels.sum() * 100
    accuracy = true_positives / 2 + true_negatives / 2

    return true_positives, true_negatives, false_positives, false_negatives, accuracy

def log_data(space, loss, pred, labels):
    true_positives, true_negatives, false_positives, false_negatives, acc = calculate_metrics(pred, labels)

    data = {f'{space}/Loss ': loss.item(),
            f'{space}/True Positives': true_positives,
            f'{space}/True Negatives': true_negatives,
            f'{space}/False Positives': false_positives,
            f'{space}/False Negatives': false_negatives,
            f'{space}/Accuracy': acc,
            f'{space}/Prediction Histogram': wandb.Histogram(torch.nn.Sigmoid()(pred).cpu().numpy())
            }

    return data

def log_input_data(morph, pose, label):
    data = {}
    data[f'input/morph_alpha'] = wandb.Histogram(morph[..., 0].cpu().numpy())
    data[f'input/morph_a'] = wandb.Histogram(morph[..., 1].cpu().numpy())
    data[f'input/morph_d'] = wandb.Histogram(morph[..., 2].cpu().numpy())
    data[f'input/translation_norm'] = wandb.Histogram(pose[:, :3].norm(dim=1).cpu().numpy())
    axis_angle = rotation_matrix_to_rotation_vector(continuous_to_rotation_matrix(pose[:, 3:]))
    data[f'input/rotation_angle0'] = wandb.Histogram(axis_angle[:, 0].cpu().numpy())
    data[f'input/rotation_angle1'] = wandb.Histogram(axis_angle[:, 1].cpu().numpy())
    data[f'input/rotation_angle2'] = wandb.Histogram(axis_angle[:, 2].cpu().numpy())
    data[f'input/(reachable/total)'] = (label == 1).sum() / label.shape[0]
    return data

def main(epochs: int,
         batch_size: int,
         early_stopping: int,
         model_hyperparameter: dict = None,
         optimiser_hyperparameter: dict = None,
         trial: optuna.Trial = None
         ):
    if model_hyperparameter is None:
        # model_hyperparameter = {
        #     'encoder_config': {'width': 512, 'depth': 1},
        #     'decoder_config': {'width': 128, 'depth': 4}
        # }
        model_hyperparameter = {
            'encoder_config': {"nhead": 8, "dim_feedforward": 640, "dropout": 0.0},
            'decoder_config': {"n_heads": 8, "mlp_dim": 640, "dropout": 0.0},
            "latent_morph": 160, "latent_pose": 160,
            "num_encoder_blocks": 4, "num_decoder_blocks": 4
        }
    if optimiser_hyperparameter is None:
        optimiser_hyperparameter = {'lr': 6e-4}
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

    training_set = SingleDataset(Path(__file__).parent.parent / 'data' / 'train', batch_size, False)
    val_set = SingleDataset(Path(__file__).parent.parent / 'data' / 'val', batch_size, True)

    model = TransTest().to(device)
    #model = ReachabilityClassifier("transformer", **model_hyperparameter).to(device)
    optimizer = optim.Adam(model.parameters(), **optimiser_hyperparameter)

    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    wandb.watch(model, criterion=loss_function, log="all", log_graph=True)

    fix_morph, fix_pose, fix_label = val_set[0]
    fix_morph = fix_morph.to(device, non_blocking=True)
    fix_pose = fix_pose.to(device, non_blocking=True)
    fix_label = fix_label.to(device, non_blocking=True)

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
            if batch_idx % 100 == 0:
                run.log(data=log_input_data(morph, pose, label), step=step, commit=False)
            pred = model(pose, morph)
            loss = loss_function(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            val_morph, val_pose, val_label = val_set.get_random_batch()
            val_morph = val_morph.to(device, non_blocking=True)
            val_pose = val_pose.to(device, non_blocking=True)
            val_label = val_label.to(device, non_blocking=True)

            model.eval()
            with torch.no_grad():
                val_pred = model(val_pose, val_morph)
                val_loss = loss_function(val_pred, val_label)

                if batch_idx % 10 == 0:
                    fix_pred = model(fix_pose, fix_morph)
                    fix_loss = loss_function(fix_pred, fix_label)
                    run.log(data=log_data('fix', fix_loss, fix_pred, fix_label), step=step, commit=False)

                run.log(data=log_data('train', loss, pred, label), step=step, commit=False)
                run.log(data=log_data('val', val_loss, val_pred, val_label), step=step, commit=True)
            if trial is not None:
                trial.report(val_loss, step)
                if trial.should_prune():
                    run.finish()
                    raise optuna.TrialPruned()

            if val_loss < min_loss:
                min_loss = val_loss
                early_stopping_counter = 0
                if trial is None:
                    torch.save(model.state_dict(), folder / "reachability_classifier.pth")
            else:
                early_stopping_counter += 1
                if early_stopping_counter == early_stopping:
                    (print('Early Stopping'))
                    run.finish()
                    return min_loss
            if batch_idx == 500:
                break

        # preds = []
        # losses = []
        # labels = []
        # for val_morph, val_pose, val_label in tqdm(val_set, desc=f"Validation"):
        #     val_morph = val_morph.to(device, non_blocking=True)
        #     val_pose = val_pose.to(device, non_blocking=True)
        #     val_label = val_label.to(device, non_blocking=True)
        #     model.eval()
        #     with torch.no_grad():
        #         val_pred = model(val_pose, val_morph * 0)
        #         val_loss = loss_function(val_pred, val_label)
        #     preds.append(val_pred.cpu())
        #     losses.append(val_loss.item())
        #     labels.append(val_label.cpu())
        # preds = torch.cat(preds, dim=0)
        # labels = torch.cat(labels, dim=0)
        # losses = torch.tensor(losses).mean()
        # run.log(data=log_data('full_val', losses, preds, labels), step=e * len(training_set) + batch_idx + 1, commit=False)
    torch.save(model.state_dict(), folder / "reachability_classifier.pth")
    run.finish()
    return min_loss


if __name__ == '__main__':
    wandb.login(key="")
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--early_stopping", type=int, default=-1)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
