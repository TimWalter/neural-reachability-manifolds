import csv
import math
import torch
from data_sampling.se3_cells import N_DIV_R3

n_div = N_DIV_R3

indices = torch.cartesian_prod(*[torch.arange(n_div)] * 3)
centres = ((indices + 0.5) / n_div) * 2.0 - 1.0
centres = centres[(centres ** 2).sum(dim=1) <= 1.0+2/N_DIV_R3*math.sqrt(3)]

with open('r3_cells.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(centres.tolist())
torch.save(centres, "r3_cells.pt")
