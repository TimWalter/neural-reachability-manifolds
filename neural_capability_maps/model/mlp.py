import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from neural_capability_maps.dataset.kinematics import transformation_matrix
from neural_capability_maps.model import Model
from neural_capability_maps.model.occupancy_network import Encoder


class MLP(Model):
    """
    Baseline for reachability prediction by fitting a simple MLP using the same encoder as the
    OccupancyNetwork.
    """

    def __init__(self, encoder_config: dict, decoder_config: dict, fourier_config: dict):
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.fourier = FourierFeatures(**fourier_config)
        self.decoder = Decoder(dim_pose_encoding=2*fourier_config["dim_encoding"],
                               dim_morph_encoding=encoder_config["dim_encoding"], **decoder_config)

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch"]:
        morph_enc = self.encoder(morph)
        #pose_enc = self.fourier(pose)
        logit = self.decoder(pose, morph_enc)
        return logit

class FourierFeatures(nn.Module):
    """
    Implements Gaussian Fourier feature mapping:
    gamma(v) = [cos(2*pi*Bv), sin(2*pi*Bv)] where B ~ N(0, sigma^2).
    """
    def __init__(self, dim_encoding: int=160, std: float = 0.167):
        super().__init__()
        b_matrix = torch.randn(dim_encoding, 9) * std
        self.register_buffer("B", b_matrix)

    def forward(self, x: Tensor) -> Tensor:
        proj = (2.0 * torch.pi * x) @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class Decoder(nn.Module):
    def __init__(self, dim_hidden: int = 1792,
                 n_blocks: int = 8,
                 dim_pose_encoding: int = 9,
                 dim_morph_encoding: int = 128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_pose_encoding + dim_morph_encoding, dim_hidden),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU())
              for _ in range(n_blocks)],
            nn.Linear(dim_hidden, 1)
        )

    def forward(self,
                pose_enc: Float[Tensor, "batch dim_pose_encoding"],
                morph_enc: Float[Tensor, "batch dim_morph_encoding"]) -> Float[Tensor, "batch"]:
        x = torch.cat([pose_enc, morph_enc], dim=-1)
        return self.model(x).squeeze(-1)