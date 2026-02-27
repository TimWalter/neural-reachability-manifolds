import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from nrm.model import Model


class MLP(Model):
    """
    Baseline for reachability prediction by fitting a simple MLP using the same encoder as the
    OccupancyNetwork.
    """

    def __init__(self, encoder_config: dict, decoder_config: dict, **kwargs):
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(dim_encoding=self.encoder.dim_encoding, **decoder_config)

    def forward(self, morph: Float[Tensor, "batch seq 3"], pose: Float[Tensor, "batch 9"]) -> Float[Tensor, "batch"]:
        morph_enc = self.encoder(morph)
        logit = self.decoder(morph_enc, pose)
        return logit


class Encoder(nn.Module):
    def __init__(self, dim_encoding: int = 512, num_layers: int = 1, drop_prob: float = 0.0):
        super().__init__()
        self.dim_encoding = dim_encoding
        self.lstm = nn.LSTM(3, dim_encoding, num_layers, dropout=drop_prob, batch_first=True, bias=False)

    def forward(self, morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch dim_encoding"]:
        if morph.is_nested:
            lengths = torch.tensor([t.size(0) for t in morph.unbind()], device=morph.device)
            morph = torch.nested.to_padded_tensor(morph, 0.0)
            morph = nn.utils.rnn.pack_padded_sequence(morph, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(morph)
        return h[-1]


class Decoder(nn.Module):
    def __init__(self, dim_hidden: int = 1792, n_blocks: int = 8, dim_encoding: int = 128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(9 + dim_encoding, dim_hidden),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU())
              for _ in range(n_blocks)],
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, morph_enc: Float[Tensor, "batch dim_encoding"], pose: Float[Tensor, "batch 9"]) \
            -> Float[Tensor, "batch"]:
        x = torch.cat([pose, morph_enc], dim=-1)
        return self.model(x).squeeze(-1)
