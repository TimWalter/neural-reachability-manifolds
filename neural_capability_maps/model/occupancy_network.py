import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from neural_capability_maps.model import Model


class OccupancyNetwork(Model):
    def __init__(self, encoder_config: dict, decoder_config: dict, **kwargs):
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(dim_encoding=encoder_config["dim_encoding"], **decoder_config)

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch"]:
        morph_enc = self.encoder(morph)
        logit = self.decoder(pose, morph_enc)
        return logit


class Encoder(nn.Module):
    def __init__(self, dim_encoding: int = 512, num_layers: int = 1, drop_prob: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(3, dim_encoding, num_layers, dropout=drop_prob, batch_first=True)

    def forward(self, morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch dim_encoding"]:
        if morph.is_nested:
            lengths = torch.tensor([t.size(0) for t in morph.unbind()], device=morph.device)
            morph = torch.nested.to_padded_tensor(morph, 0.0)
            morph = nn.utils.rnn.pack_padded_sequence(morph, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(morph)
        return h[-1]


class Decoder(nn.Module):
    def __init__(self, dim_hidden: int = 384, n_blocks: int = 8,
                 dim_encoding: int = 128,):
        super().__init__()

        self.pose_proj = nn.Conv1d(9, dim_hidden, 1)
        self.blocks = nn.ModuleList([ConditionalResnetBlock(dim_encoding, dim_hidden) for _ in range(n_blocks)])
        self.head = ConditionalConvBlock(dim_encoding, dim_hidden, 1)

        nn.init.zeros_(self.head.conv.weight)
        nn.init.zeros_(self.head.conv.bias)

    def forward(self, pose: Float[Tensor, "batch 9"], morph_enc: Float[Tensor, "batch dim_encoding"]) \
            -> Float[Tensor, "batch"]:
        pose_enc = self.pose_proj(pose.unsqueeze(-1))
        for block in self.blocks:
            pose_enc = block(pose_enc, morph_enc)
        logit = self.head(pose_enc, morph_enc)
        return logit.squeeze((1,2))


class ConditionalResnetBlock(nn.Module):
    def __init__(self, dim_encoding: int, dim_io: int):
        super().__init__()
        self.blocks = nn.ModuleList([ConditionalConvBlock(dim_encoding, dim_io, dim_io) for _ in range(2)])

        nn.init.zeros_(self.blocks[-1].conv.weight)
        nn.init.zeros_(self.blocks[-1].conv.bias)

    def forward(self, pose_enc: Float[Tensor, "batch dim_io 1"], morph_enc: Float[Tensor, "batch dim_encoding"]) \
            -> Float[Tensor, "batch dim_io 1"]:
        residual = pose_enc
        for block in self.blocks:
            pose_enc = block(pose_enc, morph_enc)
        return pose_enc + residual


class ConditionalConvBlock(nn.Module):
    def __init__(self, dim_encoding: int, dim_in: int, dim_out: int):
        super().__init__()
        self.bn = ConditionalBatchNorm(dim_encoding, dim_in)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(dim_in, dim_out, 1)

    def forward(self, pose_enc: Float[Tensor, "batch dim_in 1"], morph_enc: Float[Tensor, "batch dim_encoding"]) \
            -> Float[Tensor, "batch dim_out 1"]:
        return self.conv(self.act(self.bn(pose_enc, morph_enc)))


class ConditionalBatchNorm(nn.Module):
    def __init__(self, dim_encoding: int, dim_io: int):
        super().__init__()

        self.bn = nn.BatchNorm1d(dim_io, affine=False)
        self.mlp = nn.Linear(dim_encoding, 2 * dim_io)

        nn.init.zeros_(self.mlp.weight)
        nn.init.constant_(self.mlp.bias[:dim_io], 1.0)
        nn.init.constant_(self.mlp.bias[dim_io:], 0.0)

    def forward(self, pose_enc: Float[Tensor, "batch dim_io 1"], morph_enc: Float[Tensor, "batch dim_encoding"]) \
            -> Float[Tensor, "batch dim_io"]:
        proj = self.mlp(morph_enc).unsqueeze(-1)
        gamma, beta = proj.chunk(2, dim=1)

        return gamma * self.bn(pose_enc) + beta
