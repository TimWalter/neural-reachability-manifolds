from pathlib import Path

import torch

R3_CELLS = torch.load(Path(__file__).parent / "r3_cells.pt", map_location="cpu")
distances = torch.norm(R3_CELLS.unsqueeze(1) - R3_CELLS.unsqueeze(0), dim=-1)
R3_NEIGHBOURS = distances.argsort(dim=-1)[:, 1:7]  # Exclude self (first column)
torch.save(R3_NEIGHBOURS, Path(__file__).parent / "r3_cell_neighbours.pt")