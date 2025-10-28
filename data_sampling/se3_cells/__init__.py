from pathlib import Path
import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int

from data_sampling.orientation_representations import rotation_matrix_to_rotation_vector

R3_CELLS = torch.load(Path(__file__).parent / "r3_cells.pt", map_location="cpu")
R3_LOOKUP = torch.load(Path(__file__).parent / "r3_cell_lookup.pt", map_location="cuda")
R3_LOOKUP_MIN_DISTANCE = 2 / 18

SO3_CELLS = torch.load(Path(__file__).parent / "so3_cells.pt", map_location="cpu")
SO3_LOOKUP = torch.load(Path(__file__).parent / "so3_cell_lookup.pt", map_location="cuda")
SO3_LOOKUP_MIN_DISTANCE = 2 * torch.pi / 256

SE3_CELLS = torch.eye(4, device="cpu").repeat(R3_CELLS.shape[0] * SO3_CELLS.shape[0], 1, 1)
SE3_CELLS[:, :3, 3] = R3_CELLS.repeat_interleave(SO3_CELLS.shape[0], dim=0)
SE3_CELLS[:, :3, :3] = SO3_CELLS.repeat(R3_CELLS.shape[0], 1, 1)
N_CELLS = SE3_CELLS.shape[0]

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