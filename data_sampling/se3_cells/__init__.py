from pathlib import Path

import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int

from data_sampling.orientation_representations import rotation_matrix_to_rotation_vector

N_DIV_R3 = 18 # Has to be equal for lookup and cells!

R3_CELLS = torch.load(Path(__file__).parent / "r3_cells.pt", map_location="cpu")
N_R3 = R3_CELLS.shape[0]
R3_LOOKUP = torch.load(Path(__file__).parent / "r3_cell_lookup.pt", map_location="cuda")
R3_NEIGHBOURS = torch.load(Path(__file__).parent / "r3_cell_neighbours.pt", map_location="cpu")
R3_MAX_DISTANCE_BETWEEN_CELLS = 2 / 18

SO3_CELLS = torch.load(Path(__file__).parent / "so3_cells.pt", map_location="cpu")
N_SO3 = SO3_CELLS.shape[0]
SO3_LOOKUP = torch.load(Path(__file__).parent / "so3_cell_lookup.pt", map_location="cuda")
SO3_NEIGHBOURS = torch.load(Path(__file__).parent / "so3_cell_neighbours.pt", map_location="cpu")
SO3_MAX_DISTANCE_BETWEEN_CELLS = 0.3308

SE3_CELLS = torch.eye(4, device="cpu").repeat(R3_CELLS.shape[0] * SO3_CELLS.shape[0], 1, 1)
SE3_CELLS[:, :3, 3] = R3_CELLS.repeat_interleave(SO3_CELLS.shape[0], dim=0)
SE3_CELLS[:, :3, :3] = SO3_CELLS.repeat(R3_CELLS.shape[0], 1, 1)
SE3_MAX_DISTANCE_BETWEEN_CELLS = 0.0595

N_CELLS = SE3_CELLS.shape[0]

r3_idx = torch.arange(N_R3).repeat_interleave(N_SO3)
so3_idx = torch.arange(N_SO3).repeat(N_R3)
t_nb_r3 = R3_NEIGHBOURS[r3_idx]
t_nb_se3 = t_nb_r3 * N_SO3 + so3_idx[:, None]
so3_nb = SO3_NEIGHBOURS[so3_idx]
r_nb_se3 = r3_idx[:, None] * N_SO3 + so3_nb
SE3_NEIGHBOURS = torch.cat([t_nb_se3, r_nb_se3], dim=1)

"""
from data_sampling.robotics import LINK_RADIUS
# Mask out cells that are too close to the origin in translation and pointing towards the origin in rotation,
# which makes them theoretically unreachable
r = torch.linalg.norm(SE3_CELLS[:, :3, 3], dim=1, keepdim=True)
d = (SE3_CELLS[:, :3, 2] * SE3_CELLS[:, :3, 3] / r).sum(dim=1)
r_inner = 1 - torch.sqrt(torch.tensor([8.0]) * LINK_RADIUS ** 2)  # Pointing inwards is no longer possible
r_outer = 1 - 2 * LINK_RADIUS  # Pointing at right angle to the inward direction is no longer possible
# Stupid linear interpolation between the two results
d_max = 1 - (r - r_inner) / (r_outer - r_inner)
d_max = torch.clamp(d_max, min=0, max=1)
mask = (d > d_max.squeeze())
SE3_CELLS = SE3_CELLS[~mask]
"""



@jaxtyped(typechecker=beartype)
def pose_distance(x1: Float[Tensor, "*batch 4 4"],
                  x2: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 1"]:
    r"""
    Pose distance arising from the unique left-invariant riemannian metric for SE(3) that produces physically meaningful
    accelerations plus a weighting between translation and rotation.

    Args:
        x1: first set of homogeneous transformations.
        x2: second set of homogeneous transformations

    Returns:
        SE(3) distance between x1 and x2.

    Notes:
        Since the maximum rotational distance is \pi and the maximum translational distance in our setting is 2,
        we weigh the rotational distance by \frac{2}{\pi} for "equal" importance.
    """
    t1 = x1[..., :3, 3]
    r1 = x1[..., :3, :3]
    t2 = x2[..., :3, 3]
    r2 = x2[..., :3, :3]
    return torch.sqrt((translational_distance(t1, t2)/4) ** 2 + (1 / (2*torch.pi) * rotational_distance(r1, r2)) ** 2)


@jaxtyped(typechecker=beartype)
def translational_distance(x1: Float[Tensor, "*batch 3"],
                           x2: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 1"]:
    """
    Euclidean distance between vectors.

    Args:
        x1: first set of vectors.
        x2: second set of vectors.

    Returns:
        Euclidean distance between vectors in x1 and x2.
    """
    return torch.norm(x1 - x2, dim=-1, keepdim=True)


@jaxtyped(typechecker=beartype)
def rotational_distance(x1: Float[Tensor, "*batch 3 3"],
                        x2: Float[Tensor, "*batch 3 3"]) -> Float[Tensor, "*batch 1"]:
    """
    Geodesic distance between rotation matrices.

    Args:
        x1: first set of rotation matrices.
        x2: second set of rotation matrices.

    Returns:
        Geodesic distance between rotation matrices in x1 and x2.
    """
    R_err = torch.matmul(x1.transpose(-1, -2), x2)
    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    rot_err = torch.arccos(cos_angle)
    return rot_err.unsqueeze(-1)


@jaxtyped(typechecker=beartype)
def r3_indices(positions: Float[Tensor, "batch 3"]) -> Int[Tensor, "batch"]:
    """
    Get R3 cell indices for given positions.

    Args:
        positions: Positions in R3

    Returns:
        R3 cell indices
    """
    indices = torch.floor((positions + 1) / 2 * R3_LOOKUP.shape[0]).to(torch.int32)
    indices = torch.clamp(indices, 0, R3_LOOKUP.shape[0] - 1)  # Against numerical instability
    return R3_LOOKUP[indices[:, 0], indices[:, 1], indices[:, 2]]


@jaxtyped(typechecker=beartype)
def so3_indices(orientations: Float[Tensor, "batch 3 3"]) -> Int[Tensor, "batch"]:
    """
    Get SO3 cell indices for given orientations.

    Args:
        orientations: Orientations in SO3

    Returns:
        SO3 cell indices
    """
    rotation_vector = rotation_matrix_to_rotation_vector(orientations)

    indices = torch.floor((rotation_vector + torch.pi) / (2 * torch.pi) * SO3_LOOKUP.shape[0]).to(torch.int32)
    indices = torch.clamp(indices, 0, SO3_LOOKUP.shape[0] - 1)  # Against numerical instability
    return SO3_LOOKUP[indices[:, 0], indices[:, 1], indices[:, 2]]


@jaxtyped(typechecker=beartype)
def se3_indices(poses: Float[Tensor, "batch 4 4"]) -> Int[Tensor, "batch"]:
    """
    Get combined R3 and SO3 cell indices for given poses.

    Args:
        poses: Poses in SE(3)
    Returns:
        SE(3) cell indices
    """
    r3_index = r3_indices(poses[:, :3, 3])
    so3_index = so3_indices(poses[:, :3, :3])
    se3_index = r3_index * SO3_CELLS.shape[0] + so3_index
    return se3_index
