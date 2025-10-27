"""
Adapted based on the repo of Occupancy Network
https://github.com/autonomousvision/occupancy_networks 
"""


import torch.nn as nn
from .layers import CResnetBlockConv1d, CBatchNorm1d


class Decoder(nn.Module):
    def __init__(self,  width:int, depth:int, encoding_dim):
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

        return out.squeeze()