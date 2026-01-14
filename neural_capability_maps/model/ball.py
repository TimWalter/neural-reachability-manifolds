import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from neural_capability_maps.model import Model


class Ball(Model):
    """
    Baseline for reachability prediction by fitting a ball in R3.

    Parameters:
        - centre: Centroid of the ball.
        - radius: Radius of the ball.
        - temperature: Controls the "hardness" of the boundary.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.centre = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.radius = nn.Parameter(torch.tensor(0.75))

    def forward(self, pose: Float[Tensor, "batch 9"], _: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch 1"]:
        distance = torch.norm(pose[:, :3] - self.centre, dim=1, keepdim=True)
        logits = self.radius - distance

        return logits
