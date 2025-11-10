import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class ReachabilityClassifierTransformer(nn.Module):
    def __init__(self, latent_morph: int, encoder_config: dict, num_encoder_blocks: int,
                 latent_pose: int, decoder_config: dict, num_decoder_blocks: int):
        assert latent_morph % 2 == 0, "latent_morph must be even for rotary embeddings"
        super().__init__()
        self.morph_proj = nn.Sequential(nn.Linear(3, latent_morph), nn.ReLU())
        self.encoder = nn.Sequential(nn.TransformerEncoder(
            RotaryTransformerEncoderLayer(latent_morph, **encoder_config),
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

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) \
            -> Float[Tensor, "batch 1"]:
        morph_latent = self.morph_proj(morph)
        morph_enc = self.encoder(morph_latent)

        x = self.pose_proj(pose)
        for block in self.decoder:
            x = block(x, morph_enc)

        pred = self.classifier_head(x)
        return pred


class RotaryTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, max_seq_len=2048, base=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("freq_cos", None, persistent=False)
        self.register_buffer("freq_sin", None, persistent=False)

    def init_freq(self, device):
        pos = torch.arange(self.max_seq_len, dtype=torch.float32)
        freq = 1.0 / (self.base ** (torch.arange(0, self.self_attn.embed_dim, 2).float() / self.self_attn.embed_dim))
        angles = torch.outer(pos, freq)
        self.freq_cos = angles.cos().to(device)
        self.freq_sin = angles.sin().to(device)

    def apply_rotary_embedding(self, q, k):
        max_len = q.size(1)

        cos = self.freq_cos[:max_len].unsqueeze(0)
        sin = self.freq_sin[:max_len].unsqueeze(0)
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]
        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rot, k_rot

    def forward(self, morph_latent: Float[Tensor, "batch seq latent_morph"], *args, **kwargs) \
            -> Float[Tensor, "batch seq latent_morph"]:
        if morph_latent.is_nested: # Not supported for now
            morph_latent = torch.nested.to_padded_tensor(morph_latent, 0.0)

        if self.freq_cos is None:
            self.init_freq(morph_latent.device)

        qkv = nn.functional.linear(self.norm1(morph_latent), self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q, k = self.apply_rotary_embedding(q, k)

        attn_out, _ = self.self_attn(q, k, v, need_weights=False)
        x = morph_latent + self.dropout1(attn_out)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
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
        self.output_func = nn.Sigmoid()

    def forward(self, pose: Float[Tensor, "batch 9"], morph: Float[Tensor, "batch seq 3"]) \
            -> Float[Tensor, "batch 1"]:
        if morph.is_nested:
            morph = torch.nested.to_padded_tensor(morph, 0.0)
        morph_enc = self.encoder(morph)
        output = self.decoder(pose, morph_enc)
        return self.output_func(output)


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
