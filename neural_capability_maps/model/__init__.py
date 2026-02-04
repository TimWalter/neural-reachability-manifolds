import json
import re
from pathlib import Path

import torch
import torch.nn as nn


class Model(nn.Module):
    @classmethod
    def from_id(cls, model_id: int):
        """
        Instantiate a model from its wandb ID.

        Args:
            model_id: Wandb ID of the model.

        Returns:
            model: Instantiated model.
        """

        model_dir = Path(__file__).parent.parent.parent / "trained_models"
        pattern = rf"{model_id}-[a-z]+-[a-z]+"
        folder = next((f for f in model_dir.iterdir() if re.match(pattern, f.name)), None)
        metadata_path = model_dir / folder / 'metadata.json'
        metadata = json.load(open(metadata_path, 'r'))

        model = cls(**metadata["hyperparameter"])
        model_folder = str(model_dir / folder)
        model.load_state_dict(torch.load(next(Path(model_folder).glob('*.pth'))))
        return model

    @torch.inference_mode()
    def predict(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


from .occupancy_network import OccupancyNetwork
from .torus import Torus
from .shell import Shell
from .mlp import MLP