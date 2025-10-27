import math

import torch
from torch import Tensor
from jaxtyping import Int, Float


def index_to_r3(idx: Int[Tensor, "batch_dim 3"], n_div: int) -> Float[Tensor, "batch_dim 3"]:
    """
    Convert cell index to the center of a cube cell in [-1, 1]^3.

    Args:
        idx: the cell indices.
        n_div: Number of divisions along each axis.

    Returns:
        the center coordinates of the cube cells
    """
    return ((idx + 0.5) / n_div) * 2.0 - 1.0


def euclidean_distance(x1: Float[Tensor, "batch_dim1 3"],
                       x2: Float[Tensor, "batch_dim2 3"]) -> Float[Tensor, "batch_dim1 batch_dim2"]:
    """
    Euclidean distance between vectors.

    Args:
        x1: first set of vectors.
        x2: second set of vectors.

    Returns:
        pairwise Euclidean distance between vectors in x1 and x2.
    """
    diff = x1[:, None, :] - x2[None, :, :]
    return torch.norm(diff, dim=2)


r3_cells = torch.load("r3_cells.pt")
n_div = math.ceil(r3_cells.shape[0] ** (1/3))


indices = torch.cartesian_prod(*[torch.arange(n_div)] * 3)
cube_cells = index_to_r3(indices, n_div)
dists = euclidean_distance(cube_cells, r3_cells)
nearest_idx = torch.argmin(dists, dim=1)
cube = nearest_idx.reshape(n_div, n_div, n_div).to(torch.int32)

torch.save(cube, "r3_cell_lookup.pt")
