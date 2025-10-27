import torch
import pandas as pd

n_div = 18

indices = torch.cartesian_prod(*[torch.arange(n_div)] * 3)

centres = ((indices + 0.5) / n_div) * 2.0 - 1.0

centres = centres[(centres ** 2).sum(dim=1) <= 1.0]

pd.DataFrame(centres.numpy()).to_csv("r3_cells.csv", index=False, header=False)
torch.save(centres, "r3_cells.pt")
