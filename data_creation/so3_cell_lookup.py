import torch
from torch import Tensor
from jaxtyping import Int, Float
from scipy.spatial.transform import Rotation


def index_to_quaternion(idx: Int[Tensor, "batch_dim 3"], n_div: int) -> Float[Tensor, "batch_dim 4"]:
    """Convert cell index to the center quaternion using Euler angles.

    Args:
        idx: cell indices
        n_div: number of divisions along each axis
    Returns:
        quaternions
    """
    rotation_vector = ((idx + 0.5) / n_div) * 2 * torch.pi - torch.pi
    return torch.from_numpy(Rotation.from_rotvec(rotation_vector).as_quat()).to(torch.float32)


def quaternion_distance(q1: Float[Tensor, "batch_dim1 4"], q2: Float[Tensor, "batch_dim2 4"]) \
        -> Float[Tensor, "batch_dim1 batch_dim2"]:
    """
    Angular distance between quaternions.
    Args:
        q1: quaternions 1
        q2: quaternions 2
    Returns:
        Angular distance between each pair of quaternions in q1 and q2.
    """
    return 2 * torch.arccos(torch.abs(q1 @ q2.T).clamp(-1.0, 1.0))


n_div = 256

so3_cells = torch.load("so3_cells.pt")

indices = torch.cartesian_prod(*[torch.arange(n_div)] * 3)
cube_cells = index_to_quaternion(indices, n_div)
dists = quaternion_distance(cube_cells, so3_cells)
nearest_idx = torch.argmin(dists, dim=1)
cube = nearest_idx.reshape(n_div, n_div, n_div).to(torch.int32)

torch.save(cube, "so3_cell_lookup.pt")
