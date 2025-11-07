from typing import Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F


# === Rotary embedding utilities ===
def build_rotary_freqs(dim, max_seq_len=2048, base=10000):
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.einsum('i,j->ij', pos, freqs)
    freqs_cos = angles.cos()
    freqs_sin = angles.sin()
    return freqs_cos, freqs_sin

def apply_rotary(q, k, freqs_cos, freqs_sin):
    # q, k: (..., seq_len, dim)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.cat([q1 * freqs_cos - q2 * freqs_sin,
                       q1 * freqs_sin + q2 * freqs_cos], dim=-1)
    k_rot = torch.cat([k1 * freqs_cos - k2 * freqs_sin,
                       k1 * freqs_sin + k2 * freqs_cos], dim=-1)
    return q_rot, k_rot


# === Rotary-enhanced TransformerEncoderLayer ===
class RotaryTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, max_seq_len=2048, base=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("freqs_cos", None, persistent=False)
        self.register_buffer("freqs_sin", None, persistent=False)

    def _maybe_init_freqs(self, seq_len, device):
        if self.freqs_cos is None or seq_len > self.freqs_cos.size(0):
            freqs_cos, freqs_sin = build_rotary_freqs(self.self_attn.embed_dim, seq_len, self.base)
            self.freqs_cos = freqs_cos.to(device)
            self.freqs_sin = freqs_sin.to(device)

    def forward(self, src, *args, **kwargs):
        """Batch-first input; no causal mask."""
        seq_len = src.size(1)
        self._maybe_init_freqs(seq_len, src.device)

        x = src
        x_norm = self.norm1(x)

        # manual QKV projection
        qkv = F.linear(x_norm, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # apply rotary embeddings to Q,K
        q, k = apply_rotary(q, k, self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])

        # run standard PyTorch MHA (uses fused kernel if possible)
        attn_output, _ = self.self_attn(query=q, key=k, value=v, need_weights=False)

        # standard residual + FFN
        x = self.norm2(x + self.dropout1(attn_output))
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        return x

class ReachabilityClassifierTransformer(nn.Module):
    def __init__(self,
                 latent_morph: int, encoder_config: dict, num_encoder_blocks: int,
                 latent_pose: int, decoder_config: dict, num_decoder_blocks: int):
        super().__init__()
        self.morph_proj = nn.Sequential(nn.Linear(3, latent_morph), nn.ReLU())
        self.encoder = nn.Sequential(
            nn.TransformerEncoder(RotaryTransformerEncoderLayer(d_model=latent_morph, batch_first=True, **encoder_config),
                                  num_layers=num_encoder_blocks),
            nn.LayerNorm(latent_morph))
        self.pose_proj = nn.Sequential(nn.Linear(9, latent_pose), nn.ReLU())
        self.decoder = nn.ModuleList([PoseDecoder(latent_pose, **decoder_config)
                                       for _ in range(num_decoder_blocks)])
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(latent_pose),
            nn.Linear(latent_pose, 1),
            nn.Sigmoid()
        )

    def forward(self,  pose: Float[Tensor, "batch 9"], morphology: Float[Tensor, "batch seq 3"]) \
            -> Float[Tensor, "batch 1"]:
        morph_latent = self.morph_proj(morphology)
        morph_enc = self.encoder(morph_latent)
        pose_latent = self.pose_proj(pose)
        pose_dec = pose_latent
        for block in self.decoder:
            pose_dec = block(pose_dec, morph_enc)
        out = self.classifier_head(pose_dec)
        return out


class PoseDecoder(nn.Module):
    def __init__(self, latent_pose: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_pose, num_heads=n_heads, batch_first=True,
                                                dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_pose)
        self.ff = nn.Sequential(
            nn.Linear(latent_pose, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, latent_pose)
        )
        self.norm2 = nn.LayerNorm(latent_pose)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pose_latent: Float[Tensor, "batch latent_pose"],
                morph_enc: Float[Tensor, "batch seq latent_pose"]) -> Float[Tensor, "batch latent_pose"]:
        q = pose_latent.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=q, key=morph_enc, value=morph_enc)
        x = self.norm1(q + self.dropout(attn_out))
        x2 = self.ff(x)
        x = self.norm2(x + self.dropout(x2))
        return x.squeeze(1)


class ReachabilityClassifier(nn.Module):

    def __init__(self, encoder_config: dict, decoder_config: dict):
        super().__init__()
        # Does initialization with 0 make sense in this scenario? such that it predicts all unreachable initially
        self.output_func = nn.Sigmoid()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(encoding_dim=encoder_config["width"], **decoder_config)

    def forward(self, poses, morph):
        morph_encodings = self.encoder(morph)
        output = self.decoder(poses, morph_encodings)
        return self.output_func(output)


class Encoder(nn.Module):
    def __init__(self, width: int, depth: int, drop_prob=0):
        super().__init__()
        self.width = width
        self.depth = depth
        self.lstm = nn.LSTM(3, width, depth, dropout=drop_prob, batch_first=True)

    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.is_nested:
            # Convert nested tensor to padded tensor
            x_padded = torch.nested.to_padded_tensor(x, padding=0.0)

            # Extract actual lengths from nested tensor
            # The nested tensor stores the actual sequence lengths
            lengths = torch.tensor([t.size(0) for t in x.unbind()], device=x.device)

            # Now proceed with the padded tensor
            batch_size = x_padded.size(0)
            h0 = torch.zeros(self.depth, batch_size, self.width).to(x.device)
            c0 = torch.zeros(self.depth, batch_size, self.width).to(x.device)

            # Pack the padded sequence with actual lengths
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (final_h, _) = self.lstm(packed_x, (h0, c0))
            return final_h[-1]
        else:
            # Original padded tensor path
            h0 = torch.zeros(self.depth, x.size(0), self.width).to(x.device)
            c0 = torch.zeros(self.depth, x.size(0), self.width).to(x.device)
            lengths = torch.ones(x.shape[0], device="cpu") * (x.shape[1] - 1)
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (final_h, _) = self.lstm(packed_x, (h0, c0))
            return final_h[-1]


class Decoder(nn.Module):
    def __init__(self, width: int, depth: int, encoding_dim):
        super().__init__()

        self.conv_p = nn.Conv1d(9, width, 1)
        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(encoding_dim, width) for _ in range(depth)
        ])

        self.bn = CBatchNorm1d(encoding_dim, width)
        self.conv_out = nn.Conv1d(width, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, c):
        net = self.conv_p(p.unsqueeze(2))

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))

        return out.squeeze(2)


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm'):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = CBatchNorm1d(
            c_dim, size_in, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(
            c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert (x.size(0) == c.size(0))
        assert (c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out
