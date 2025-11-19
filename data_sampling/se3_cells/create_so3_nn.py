from pathlib import Path

import torch
from data_sampling.se3_cells import rotational_distance

SO3_CELLS = torch.load(Path(__file__).parent / "so3_cells.pt", map_location="cpu")
distances = rotational_distance(SO3_CELLS.unsqueeze(0).expand(SO3_CELLS.shape[0], SO3_CELLS.shape[0], 3, 3),
                                SO3_CELLS.unsqueeze(1).expand(SO3_CELLS.shape[0], SO3_CELLS.shape[0], 3, 3)).squeeze(-1)
SO3_NEIGHBOURS = distances.argsort(dim=-1)[:, 1:7]  # Exclude self (first column)
torch.save(SO3_NEIGHBOURS, Path(__file__).parent / "so3_cell_neighbours.pt")
