import json
from typing import Type
from pathlib import Path

import optuna
import wandb
import torch
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from jaxtyping import Float, Bool
from torch import Tensor

import neural_capability_maps.dataset.se3 as se3
from neural_capability_maps.dataset.capability_map import estimate_reachable_ball
from neural_capability_maps.visualisation import get_pose_traces
from neural_capability_maps.model import Model
from neural_capability_maps.dataset.loader import TrainingSet, ValidationSet


class Logger:
    def __init__(self, model_class: Type[Model],
                 hyperparameter: dict,
                 epochs: int,
                 batch_size: int,
                 early_stopping: int,
                 lr: float,
                 trial: optuna.Trial,
                 training_set: TrainingSet,
                 validation_set: ValidationSet,
                 model: Model,
                 loss_function: torch.nn.modules.loss._Loss
                 ):
        self.batch_size = batch_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_function = loss_function
        metadata = {"num_training_samples": len(training_set) * batch_size,
                    "num_validation_samples": len(validation_set) * batch_size,
                    "model_class": model_class.__name__,
                    "hyperparameter": hyperparameter,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "early_stopping": early_stopping,
                    "lr": lr}

        self.run = self.setup_wandb(metadata, trial)
        self.folder = self.save_metadata(metadata)

        self.fix = list(validation_set.get_random_batch())
        self.fix = [f.to(self.device, non_blocking=True) for f in self.fix]

        self.buffer = {}

        self.step = 0

    def setup_wandb(self, metadata: dict, trial: optuna.Trial) -> wandb.Run:
        wandb.login(key="")
        run = wandb.init(project="Capability Maps",
                         config=metadata,
                         group="Reachability",
                         tags=[metadata["model_class"]],
                         dir=Path(__file__).parent.parent / "wandb")

        if trial is not None:
            run.name = f"trial/{trial.number}/{run.name}"
        #wandb.watch(self.model, criterion=self.loss_function, log="all")

        return run

    def save_metadata(self, metadata: dict) -> Path:
        parts = self.run.name.split("-")
        folder = Path(__file__).parent.parent / "trained_models" / f"{parts[-1]}-{'-'.join(parts[:-1])}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        json.dump(metadata, open(folder / 'metadata.json', 'w'), indent=4)
        return folder

    def save_model(self):
        torch.save(self.model.state_dict(), self.folder / "checkpoint.pth")

    def __del__(self):
        self.run.finish()

    @staticmethod
    def compute_metrics(logit: Float[Tensor, "batch"], label: Bool[Tensor, "batch"]) -> dict:
        (true_positives, false_negatives), (false_positives, true_negatives) = binary_confusion_matrix(logit, label)
        accuracy = ((torch.nn.Sigmoid()(logit) > 0.5) == label).cpu().sum() / label.shape[0] * 100

        hist, bin_edges = np.histogram(torch.nn.Sigmoid()(logit).cpu().numpy(), bins=64, range=(0.0, 1.0))

        metrics = {'True Positives': true_positives,
                   'True Negatives': true_negatives,
                   'False Positives': false_positives,
                   'False Negatives': false_negatives,
                   'Accuracy': accuracy,
                   'F1 Score': 2 * true_positives / (2 * true_positives + false_positives + false_negatives) * 100,
                   'Predictions': wandb.Histogram(np_histogram=(hist, bin_edges)),
                   }

        return metrics

    @staticmethod
    def compute_input_metrics(morph: Float[Tensor, "batch dof 3"],
                              pose: Float[Tensor, "batch 9"],
                              label: Bool[Tensor, "batch"]) -> dict:
        metrics = {"DOF": wandb.Histogram((morph != 0).any(dim=-1).sum(dim=-1).cpu().numpy() - 1)}

        centre, radius = estimate_reachable_ball(morph)
        metrics["Length (Excluding Base)"] = torch.mean(1 - centre.norm(dim=1).cpu())

        rel_radius = (pose[:, :3] - centre).norm(dim=1) / radius
        metrics["Relative Radius"] = wandb.Histogram(rel_radius.cpu().numpy())

        fig = go.Figure()
        colors = sns.color_palette("colorblind", n_colors=2).as_hex()
        for t in get_pose_traces(morph[label][:1000].cpu(), se3.from_vector(pose[label][:1000]).cpu(), colors[0], "Reachable",
                                 True):
            fig.add_trace(t)
        for t in get_pose_traces(morph[~label][:1000].cpu(), se3.from_vector(pose[~label][:1000]).cpu(), colors[1],
                                 "Unreachable", True):
            fig.add_trace(t)
        metrics["Poses"] = fig

        metrics["Reachable [%]"] = label.sum().item() / label.shape[0] * 100

        return metrics

    @staticmethod
    def assign_space(data: dict, space: str) -> dict:
        for key in list(data.keys()):
            data[f"{space}/{key}"] = data.pop(key)
        return data

    @torch.no_grad()
    def log_training(self,
                     epoch: int,
                     batch_idx: int,
                     morph: Float[Tensor, "batch dof 3"],
                     pose: Float[Tensor, "batch 9"],
                     label: Bool[Tensor, "batch"],
                     logit: Float[Tensor, "batch"],
                     loss: float):
        data = {"Loss": loss / self.batch_size, "Batch ID": self.training_set.current_batch_idx}
        data |= self.compute_metrics(logit, label)
        data = self.assign_space(data, "Training")

        if batch_idx % 1000 == 0:
            data |= self.assign_space(self.compute_input_metrics(morph, pose, label), "TrainingsInput")

        if batch_idx % 10 == 0:
            morph, pose, label = self.validation_set.get_semi_random_batch()
            morph = morph.to(self.device, non_blocking=True)
            pose = pose.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            logit = self.model(pose, morph)
            loss = self.loss_function(logit, label.float())

            intermediate_val_data = {"Loss": loss}
            intermediate_val_data |= self.compute_metrics(logit, label)
            intermediate_val_data = self.assign_space(intermediate_val_data, "Intermediate Validation")
            data |= intermediate_val_data

        self.step = epoch * len(self.training_set) + batch_idx
        self.run.log(data=data, step=self.step, commit=True)

    @torch.no_grad()
    def log_validation(self,
                       epoch: int,
                       batch_idx: int,
                       morph: Float[Tensor, "batch dof 3"],
                       pose: Float[Tensor, "batch 9"],
                       label: Bool[Tensor, "batch"],
                       logit: Float[Tensor, "batch"],
                       loss: float):
        if "loss" not in self.buffer:
            self.buffer["loss"] = 0.0
        if "logit" not in self.buffer:
            self.buffer["logit"] = []
        if "morph" not in self.buffer:
            self.buffer["morph"] = []
        if "pose" not in self.buffer:
            self.buffer["pose"] = []
        if "label" not in self.buffer:
            self.buffer["label"] = []

        self.buffer["morph"] += [morph.cpu()]
        self.buffer["pose"] += [pose.cpu()]
        self.buffer["label"] += [label.cpu()]
        self.buffer["logit"] += [logit.cpu()]
        self.buffer["loss"] += loss

        if batch_idx + 1 == len(self.validation_set):
            morph = torch.cat(self.buffer["morph"])
            pose = torch.cat(self.buffer["pose"])
            label = torch.cat(self.buffer["label"])

            data = {"Loss": self.buffer["loss"] / len(self.validation_set) / self.batch_size}
            data |= self.compute_metrics(torch.cat(self.buffer["logit"]), label)
            data = self.assign_space(data, "Validation")

            data |= self.assign_space(self.compute_input_metrics(morph, pose, label), "ValidationInput")

            self.run.log(data=data, step=self.step+1, commit=False)
            self.buffer = {}


def binary_confusion_matrix(logit: Float[Tensor, "batch"], label: Bool[Tensor, "batch"]) \
        -> Float[Tensor, "2 2"]:
    predicted_label = torch.nn.Sigmoid()(logit) > 0.5
    confusion_matrix = torch.zeros(2, 2)
    confusion_matrix[0, 0] = (predicted_label & label).sum().item() / label.sum() * 100  # TP
    confusion_matrix[0, 1] = (~predicted_label & label).sum().item() / label.sum() * 100  # FN
    confusion_matrix[1, 0] = (predicted_label & ~label).sum().item() / (~label).sum() * 100  # FP
    confusion_matrix[1, 1] = (~predicted_label & ~label).sum().item() / (~label).sum() * 100  # TN
    return confusion_matrix
