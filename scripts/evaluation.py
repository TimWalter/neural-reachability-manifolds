import argparse
import torch
from model import Model
from data_sampling.dataset import Dataset
from tqdm import tqdm
from pathlib import Path
import json
import re
import wandb
from torcheval.metrics import BinaryConfusionMatrix, R2Score
from train import model_type_settings


def eval_log(table, loss, mdh, metric, settings, run):
    table.add_data(loss, str(mdh), *([metric.normalized("true").cpu().item()] if settings["only_reachable"] else
                                     [metric.normalized("true").cpu()[1, 1], metric.normalized("true").cpu()[0, 0]]))
    run.log({"Evaluation": table})


def main(model_type: str,
         model_folder: str,
         encoder_width: int,
         encoder_depth: int,
         decoder_width: int,
         decoder_depth: int):
    settings = model_type_settings[args.model_type]

    run = wandb.init(project="Capability Maps",
                     config={"model": model_folder},
                     group="eval_" + model_type)

    device = torch.device("cuda")


    test_set = Dataset(Path(__file__).parent.parent / 'data' / 'test', device, 100000, settings["only_reachable"], 0.0)

    model = Model(
        encoder_config={'width': encoder_width, 'depth': encoder_depth},
        decoder_config={'width': decoder_width, 'depth': decoder_depth},
        output_func=settings["activation_function"]
    ).to(device)
    model.load_state_dict(torch.load(next(Path(model_folder).glob('*.pth')), map_location=device))

    loss_function = settings["loss_function"]

    table = wandb.Table(log_mode="MUTABLE",
                        columns=["Loss", "MDH"] +
                                (["R2Score"] if settings["only_reachable"] else ["Correct Positives",
                                                                                 "Correct Negatives"]))

    current_mdh = None
    loss = 0.0
    sample_count = 0
    metric = R2Score() if settings["only_reachable"] else BinaryConfusionMatrix()

    model.eval()
    for i, (weights, mdhs, poses, labels) in enumerate(tqdm(test_set, desc=f"Evaluation")):
        if current_mdh is not None and (current_mdh != mdhs[0]).any():
            eval_log(table, loss, current_mdh, metric, settings, run)
        if current_mdh is None or (current_mdh != mdhs[0]).any():
            current_mdh = mdhs[0]
            loss = 0.0
            sample_count = 0
            metric.reset()

        # weights = weights.to(device, non_blocking=True)
        # mdhs = mdhs.to(device, non_blocking=True)
        # poses = poses.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(poses, mdhs)
        loss = (loss * sample_count + loss_function(pred, labels, weights) * labels.shape[0]) / (
                sample_count + labels.shape[0])
        sample_count += labels.shape[0]
        metric.update(pred, labels if settings["only_reachable"] else (labels != -1).to(torch.int64))

    eval_log(table, loss, current_mdh, metric, settings, run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="reachability_classifier")
    parser.add_argument("--model_id", type=int, default=388)
    args = parser.parse_args()

    model_dir = Path(__file__).parent.parent / "trained_models"
    pattern = rf"{re.escape(args.model_type)}_[a-z]+-[a-z]+-{args.model_id}"
    folder = next((f for f in model_dir.iterdir() if re.match(pattern, f.name)), None)
    path = model_dir / folder / 'settings.json'
    settings = json.load(open(path, 'r'))

    main(args.model_type,
         str(model_dir / folder),
         settings.get("encoder_width", 512),
         settings.get("encoder_depth", 1),
         settings.get("decoder_width", 128),
         settings.get("decoder_depth", 4))
