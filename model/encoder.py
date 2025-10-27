import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, width:int, depth:int, drop_prob=0):
        super().__init__()
        self.width = width
        self.depth = depth
        self.lstm = nn.LSTM(3, width, depth, dropout=drop_prob, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.depth, x.size(0), self.width).to(x.device)
        c0 = torch.zeros(self.depth, x.size(0), self.width).to(x.device)
        lengths = torch.ones(x.shape[0], device="cpu") * (x.shape[1] - 1)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h, _) = self.lstm(packed_x, (h0, c0))

        return final_h[-1]
