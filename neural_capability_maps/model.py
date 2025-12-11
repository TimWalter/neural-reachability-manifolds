import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

import numpy as np

class TransTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_proj = nn.Sequential(nn.Linear(9, 160), nn.ReLU())
        self.decoder = nn.ModuleList([
            TestDecoder(160, **{"n_heads": 8, "mlp_dim": 640, "dropout": 0.0})
                                      for _ in range(4)])
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(160),
            nn.Linear(160, 1)
        )

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) \
            -> Float[Tensor, "batch 1"]:
        x = self.pose_proj(pose)
        for block in self.decoder:
            x = block(x)
        pred = self.classifier_head(x)
        return pred

class TestDecoder(nn.Module):
    def __init__(self, latent_pose: int, n_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_pose, num_heads=n_heads,
                                                batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_pose)
        self.norm2 = nn.LayerNorm(latent_pose)
        self.mlp = nn.Sequential(
            nn.Linear(latent_pose, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, latent_pose)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pose_latent: Float[Tensor, "batch latent_pose"]) -> Float[Tensor, "batch latent_pose"]:
        q = self.norm1(pose_latent).unsqueeze(1)
        x = pose_latent.unsqueeze(1) + self.dropout(q)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x.squeeze(1)


class MLP(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=12):
        super().__init__()
        layers = [
            nn.Linear(9, hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pose: Float[Tensor, "batch 9"], _) -> Float[Tensor, "batch 1"]:
        return self.net(pose)



class SineLayer(nn.Module):
    """
    Fundamental building block for SIREN.
    Applies a linear transformation followed by sin(omega_0 * x).
    Includes specific initialization schemes for first vs hidden layers.
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer init: Uniform(-1/in, 1/in)
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                # Hidden layer init: Uniform(-sqrt(6/n)/omega, sqrt(6/n)/omega)
                # This keeps the activation distribution consistent across layers
                limit = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SingleMorphologySIREN(nn.Module):
    def __init__(self,
                 input_dim: int = 9,    # SE(3) pose (e.g., pos + rot6d or flattened matrix)
                 hidden_dim: int = 512, # 256 or 512 is standard for SDFs/Occupancy
                 num_layers: int = 10,   # Depth allows composing higher frequencies
                 output_dim: int = 1,   # 1 for Reachability probability/logit
                 omega_0: float = 60.0):
        super().__init__()

        layers = []

        # 1. First Layer (Special Init, maps coords to hidden features)
        layers.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0))

        # 2. Hidden Layers
        for _ in range(num_layers - 2): # -2 because we handle first and last separately
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))

        # 3. Output Layer
        # Standard SIRENs usually end with a Linear layer (no activation) to get logits/SDF values.
        # If you need probability 0-1 directly, apply Sigmoid here or in the loss function.
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize final layer for small output to help convergence at start
        with torch.no_grad():
            self.final_layer.weight.uniform_(-np.sqrt(6/hidden_dim) / omega_0,
                                             np.sqrt(6/hidden_dim) / omega_0)

    def forward(self, pose: Float[Tensor, "batch 9"], _) -> Float[Tensor, "batch 1"]:
        # Pass through Sine layers
        features = self.net(pose)

        # Linear projection to score
        logits = self.final_layer(features)

        return logits

class ReachabilityClassifier(nn.Module):
    def __init__(self, architecture: str, **kwargs):
        super().__init__()
        if architecture == "transformer":
            self.model = ReachabilityClassifierTransformer(**kwargs)
        elif architecture == "lstm":
            self.model = ReachabilityClassifierLSTM(**kwargs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) \
            -> Float[Tensor, "batch 1"]:
        return self.model(pose, morph)


class ReachabilityClassifierTransformer(nn.Module):
    def __init__(self, latent_morph: int, encoder_config: dict, num_encoder_blocks: int,
                 latent_pose: int, decoder_config: dict, num_decoder_blocks: int):
        assert latent_morph % 2 == 0, "latent_morph must be even for rotary embeddings"
        super().__init__()
        self.morph_proj = nn.Sequential(nn.Linear(3, latent_morph), nn.ReLU())
        self.encoder_blocks = nn.ModuleList([
            RotaryTransformerEncoderLayer(latent_morph, **encoder_config)
            for _ in range(num_encoder_blocks)
        ])
        self.encoder_norm = nn.LayerNorm(latent_morph)
        self.pose_proj = nn.Sequential(nn.Linear(9, latent_pose), nn.ReLU())
        self.decoder = nn.ModuleList([PoseDecoder(latent_pose, **decoder_config)
                                      for _ in range(num_decoder_blocks)])
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(latent_pose),
            nn.Linear(latent_pose, 1)
        )

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) \
            -> Float[Tensor, "batch 1"]:
        morph_latent = self.morph_proj(morph)
        x = morph_latent
        for block in self.encoder_blocks:
            x = block(x)
        morph_enc = self.encoder_norm(x)

        x = self.pose_proj(pose)
        for block in self.decoder:
            x = block(x, morph_enc)

        pred = self.classifier_head(x)
        return pred


class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_seq_len=2048, base=10000):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # 1. Self Attention components (replacing nn.MultiheadAttention)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 2. Feed Forward components
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()  # or nn.GELU()

        # 3. Norms and Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 4. RoPE components
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("freq_cos", None, persistent=False)
        self.register_buffer("freq_sin", None, persistent=False)

    def init_freq(self, device):
        pos = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        # Standard RoPE frequency calculation
        dim = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        freq = 1.0 / (self.base ** (dim / self.head_dim))
        angles = torch.outer(pos, freq)
        self.freq_cos = angles.cos()
        self.freq_sin = angles.sin()

    def apply_rotary_embedding(self, q, k):
        # q, k shape: [batch, seq, nhead, head_dim]
        # We need to reshape cos/sin to broadcast: [1, seq, 1, head_dim/2]
        seq_len = q.shape[1]
        if self.freq_cos is None or self.freq_cos.device != q.device:
            self.init_freq(q.device)

        cos = self.freq_cos[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.freq_sin[:seq_len].unsqueeze(0).unsqueeze(2)

        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rot, k_rot

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Pre-Norm architecture (Norm -> Attn -> Add) is generally more stable
        x = src

        # --- Attention Block ---
        residual = x
        x = self.norm1(x)

        batch, seq, _ = x.shape
        if self.freq_cos is None:
            self.init_freq(x.device)

        # Project
        q = self.q_proj(x).view(batch, seq, self.nhead, self.head_dim)
        k = self.k_proj(x).view(batch, seq, self.nhead, self.head_dim)
        v = self.v_proj(x).view(batch, seq, self.nhead, self.head_dim)

        # Apply RoPE
        q, k = self.apply_rotary_embedding(q, k)

        # Attention (transpose to [batch, nhead, seq, head_dim] for SDPA)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use specialized efficient attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                    dropout_p=self.dropout1.p if self.training else 0.0)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Residual 1
        x = residual + self.dropout1(attn_out)

        # --- Feed Forward Block ---
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Residual 2
        x = residual + self.dropout2(x)

        return x


class PoseDecoder(nn.Module):
    def __init__(self, latent_pose: int, n_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_pose, num_heads=n_heads,
                                                batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_pose)
        self.norm2 = nn.LayerNorm(latent_pose)
        self.mlp = nn.Sequential(
            nn.Linear(latent_pose, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, latent_pose)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pose_latent: Float[Tensor, "batch latent_pose"],
                morph_enc: Float[Tensor, "batch seq latent_pose"]) -> Float[Tensor, "batch latent_pose"]:
        q = self.norm1(pose_latent).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, key=morph_enc, value=morph_enc)
        x = pose_latent.unsqueeze(1) + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x.squeeze(1)


class ReachabilityClassifierLSTM(nn.Module):
    def __init__(self, encoder_config: dict, decoder_config: dict):
        super().__init__()
        self.encoder = EncoderLSTM(**encoder_config)
        self.decoder = DecoderLSTM(encoding_dim=encoder_config["width"], **decoder_config)

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) \
            -> Float[Tensor, "batch 1"]:
        if morph.is_nested:
            morph = torch.nested.to_padded_tensor(morph, 0.0)
        morph_enc = self.encoder(morph)
        output = self.decoder(pose, morph_enc)
        return output


class EncoderLSTM(nn.Module):
    def __init__(self, width: int, depth: int, drop_prob: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(3, width, depth, dropout=drop_prob, batch_first=True)

    def forward(self, morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch encoding_dim"]:
        if morph.is_nested:
            lengths = torch.tensor([t.size(0) for t in morph.unbind()], device=morph.device)
            morph = torch.nested.to_padded_tensor(morph, 0.0)
            morph = nn.utils.rnn.pack_padded_sequence(morph, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(morph)
        return h[-1]


class DecoderLSTM(nn.Module):
    def __init__(self, width: int, depth: int, encoding_dim):
        super().__init__()

        self.pose_proj = nn.Conv1d(9, width, 1)
        self.blocks = nn.ModuleList([CResnetBlockConv1d(encoding_dim, width) for _ in range(depth)])
        self.out_bn = CBatchNorm1d(encoding_dim, width)
        self.out_conv = nn.Conv1d(width, 1, 1)
        self.act = nn.ReLU()

    def forward(self, pose: Float[Tensor, "batch 9"], morph_enc: Float[Tensor, "batch encoding_dim"]) \
            -> Float[Tensor, "batch 1"]:
        x = self.pose_proj(pose.unsqueeze(2))
        for block in self.blocks:
            x = block(x, morph_enc)
        x = self.act(self.out_bn(x, morph_enc))
        x = self.out_conv(x)
        return x.squeeze(2)


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
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


def load_model(model_id: int) -> ReachabilityClassifier:
    model_dir = Path(__file__).parent.parent / "trained_models"
    pattern = rf"{re.escape("reachability_classifier")}_[a-z]+-[a-z]+-{model_id}"
    folder = next((f for f in model_dir.iterdir() if re.match(pattern, f.name)), None)
    path = model_dir / folder / 'settings.json'
    settings = json.load(open(path, 'r'))
    if "latent_morph" in settings["model_hyperparameter"]:
        architecture = "transformer"
    else:
        architecture = "lstm"

    model = ReachabilityClassifier(architecture, **settings["model_hyperparameter"]).to("cuda")
    model_folder = str(model_dir / folder)
    model.load_state_dict(torch.load(next(Path(model_folder).glob('*.pth')), map_location="cuda"))
    return model
