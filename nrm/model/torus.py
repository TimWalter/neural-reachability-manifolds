import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from nrm.dataset.kinematics import transformation_matrix
from nrm.model import Model
from nrm.model.occupancy_network import Encoder


class Torus(Model):
    """
    Baseline for reachability prediction by fitting a Torus (Doughnut) in R3 using the same encoder as the
    OccupancyNetwork.
    """

    def __init__(self, encoder_config: dict, **kwargs):
        super().__init__()

        self.encoder = Encoder(**encoder_config)
        self.head = nn.Linear(encoder_config["dim_encoding"], 2)
        self.activation = nn.Softplus()

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch"]:
        radii = self.activation(self.head(self.encoder(morph)))
        minor_radius = radii[:, 0]
        major_radius = radii[:, 1]

        mat = transformation_matrix(morph[:, 0, 0:1],
                                    morph[:, 0, 1:2],
                                    morph[:, 0, 2:3],
                                    torch.zeros_like(morph[:, 0, 0:1]))
        centre = mat[:, :3, 3]
        torus_axis = torch.nn.functional.normalize(mat[:, :3, 2], dim=1)
        position = pose[:, :3] - centre

        distance_axis = torch.sum(position * torus_axis, dim=1)
        distance_plane = torch.norm(position - distance_axis.unsqueeze(1) * torus_axis, dim=1)
        distance_core = torch.sqrt((distance_plane - major_radius) ** 2 + distance_axis ** 2)

        return minor_radius - distance_core
