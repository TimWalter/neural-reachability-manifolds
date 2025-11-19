import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Int, Float, jaxtyped

from data_sampling.se3_cells import rotational_distance
from data_sampling.orientation_representations import rotation_vector_to_rotation_matrix

@jaxtyped(typechecker=beartype)
def index_to_rotation_matrix(idx: Int[Tensor, "*batch 3"], n_div: int) -> Float[Tensor, "*batch 3 3"]:
    """Convert cell index to the center quaternion using Euler angles.

    Args:
        idx: cell indices
        n_div: number of divisions along each axis
    Returns:
        rotation matrices
    """
    rotation_vector = ((idx + 0.5) / n_div) * 2 * torch.pi - torch.pi
    return rotation_vector_to_rotation_matrix(rotation_vector)

n_div = 64

so3_cells = torch.load("so3_cells.pt")

indices = torch.cartesian_prod(*[torch.arange(n_div)] * 3)
lookup_centre = index_to_rotation_matrix(indices, n_div)
distances = rotational_distance(so3_cells.unsqueeze(0).expand(lookup_centre.shape[0], so3_cells.shape[0], 3, 3),
                                lookup_centre.unsqueeze(1).expand(lookup_centre.shape[0], so3_cells.shape[0], 3, 3)).squeeze(-1)

nearest_idx = torch.argmin(distances, dim=1)
cube = nearest_idx.reshape(n_div, n_div, n_div).to(torch.int32)

torch.save(cube, "so3_cell_lookup.pt")
