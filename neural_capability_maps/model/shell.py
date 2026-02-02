import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from neural_capability_maps.model import Model


class Shell(Model):
    """
    Baseline for reachability prediction by fitting a shell in R3.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.centre = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.inner_radius = nn.Parameter(torch.tensor(0.25))
        self.half_thickness = nn.Parameter(torch.tensor(0.75))

        self.softplus = nn.Softplus()

    def forward(self, pose: Float[Tensor, "batch 9"], _: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch"]:
        distance = torch.norm(pose[:, :3] - self.centre, dim=1)

        inner_radius = self.softplus(self.inner_radius)
        half_thickness = self.softplus(self.half_thickness)

        logits = half_thickness - self.softplus(distance - inner_radius - half_thickness)

        return logits
