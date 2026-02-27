import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from nrm.model import Model
from nrm.model.occupancy_network import Encoder


class Shell(Model):
    """
    Baseline for reachability prediction by fitting a shell in R3.
    """

    def __init__(self, encoder_config: dict, **kwargs):
        super().__init__()

        self.encoder = Encoder(**encoder_config)
        self.head = nn.Linear(encoder_config["dim_encoding"], 5)
        self.activation = nn.Softplus()

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch"]:
        parameters = self.head(self.encoder(morph))
        centre = parameters[:, :3]
        inner_radius = self.activation(parameters[:, 3])
        half_thickness = self.activation(parameters[:, 4])

        distance = torch.norm(pose[:, :3] - centre, dim=1)
        logits = half_thickness - self.activation(distance - inner_radius - half_thickness)

        return logits
