import re
import json
import wandb
import torch
import argparse

from model import ReachabilityClassifier
from dataset import SingleDataset
from tqdm import tqdm
from pathlib import Path


def main(model_folder: str, model_hyperparameter: dict, **kwargs):
    device = torch.device("cuda")

    run = wandb.init(project="Capability Maps", config={"model": model_folder}, group="eval_reachability_classifier")

    test_set = SingleDataset(Path(__file__).parent.parent / 'data' / 'train', 10_000, False, False)

    model = ReachabilityClassifier("lstm", **model_hyperparameter).to(device)
    model.load_state_dict(torch.load(next(Path(model_folder).glob('*.pth')), map_location=device))

    loss_function = torch.nn.BCELoss(reduction='mean')

    table = wandb.Table(columns=["Loss", "True Positives", "True Negatives", "False Positives", "False Negatives"])

    total_loss = 0.0
    sample_count = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    model.eval()
    for batch_idx, (morph, pose, label) in enumerate(tqdm(test_set, desc=f"Evaluation")):
        morph = morph.to(device, non_blocking=True)
        pose = pose.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(pose, morph)
        total_loss = (total_loss * sample_count + loss_function(pred, label).cpu() * label.shape[0]) / (
                sample_count + label.shape[0])
        sample_count += label.shape[0]
        pred_label = pred > 0.5
        true_label = label.bool()
        true_positives += (pred_label & true_label).sum().item()
        true_negatives += (~pred_label & ~true_label).sum().item()
        false_positives += (pred_label & ~true_label).sum().item()
        false_negatives += (~pred_label & true_label).sum().item()

    total_loss /= sample_count
    true_positives /= sample_count
    true_negatives /= sample_count
    false_positives /= sample_count
    false_negatives /= sample_count
    table.add_data(total_loss, true_positives, true_negatives, false_positives, false_negatives)
    run.log({"Evaluation": table})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=701)
    args = parser.parse_args()

    model_dir = Path(__file__).parent.parent / "trained_models"
    pattern = rf"{re.escape("reachability_classifier")}_[a-z]+-[a-z]+-{args.model_id}"
    folder = next((f for f in model_dir.iterdir() if re.match(pattern, f.name)), None)
    path = model_dir / folder / 'settings.json'
    settings = json.load(open(path, 'r'))

    main(**settings, model_folder=str(model_dir / folder))
