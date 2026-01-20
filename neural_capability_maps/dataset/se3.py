import math
import torch
from torch import Tensor
from jaxtyping import Float, Int64

import neural_capability_maps.dataset.r3 as r3
import neural_capability_maps.dataset.so3 as so3


# @jaxtyped(typechecker=beartype)
def distance(x1: Float[Tensor, "*batch 4 4"], x2: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 1"]:
    r"""
    Pose distance arising from the unique left-invariant riemannian metric for SE(3) that produces physically meaningful
    accelerations plus a weighting between translation and rotation.

    Args:
        x1: First homogeneous transformation.
        x2: Second homogeneous transformation.

    Returns:
        SE(3) distance between x1 and x2.

    Notes:
        Since the maximum rotational distance is \pi and the maximum translational distance in our setting is 2,
        we weigh the distances "equal" importance and also such that the maximum distance between two cells is 1.
    """
    t1 = x1[..., :3, 3]
    r1 = x1[..., :3, :3]
    t2 = x2[..., :3, 3]
    r2 = x2[..., :3, :3]
    return torch.sqrt(r3.distance(t1, t2) ** 2 / 8 + so3.distance(r1, r2) ** 2 / (2 * torch.pi ** 2))


MAX_DISTANCE_BETWEEN_CELLS = math.sqrt(r3.MAX_DISTANCE_BETWEEN_CELLS ** 2 / 8 +
                                       so3.MAX_DISTANCE_BETWEEN_CELLS ** 2 / (2 * torch.pi ** 2))
N_CELLS = r3.N_CELLS * so3.N_CELLS


# @jaxtyped(typechecker=beartype)
def split_index(index: Int64[Tensor, "*batch"]) -> tuple[Int64[Tensor, "*batch"], Int64[Tensor, "*batch"]]:
    """
    Split SE(3) cell index into R3 and SO(3) indices.

    Args:
        index: SE(3) cell index.

    Returns:
        R3 and SO(3) index
    """
    return index % r3.N_CELLS, index // r3.N_CELLS


# @jaxtyped(typechecker=beartype)
def combine_index(r3_index: Int64[Tensor, "*batch"], so3_index: Int64[Tensor, "*batch"]) -> Int64[Tensor, "*batch"]:
    """
    Combine R3 and SO(3) index into SE(3) cell index.

    Args:
        r3_index: R3 index.
        so3_index: SO(3) index.

    Returns:
        SE(3) cell index.
    """
    return r3_index + so3_index * r3.N_CELLS


# @jaxtyped(typechecker=beartype)
def index(pose: Float[Tensor, "*batch 4 4"]) -> Int64[Tensor, "*batch"]:
    """
    Get cell index for the given poses.

    Args:
        pose: Pose.

    Returns:
        Cell index.
    """
    return combine_index(r3.index(pose[:, :3, 3]), so3.index(pose[:, :3, :3]))


# @jaxtyped(typechecker=beartype)
def cell(index: Int64[Tensor, "*batch"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Get cell pose for the given index.

    Args:
        index: Cell index.

    Returns:
        Cell pose.
    """
    r3_index, so3_index = split_index(index)
    se3 = torch.eye(4, device=index.device).repeat(*index.shape, 1, 1)
    se3[..., :3, 3] = r3.cell(r3_index)
    se3[..., :3, :3] = so3.cell(so3_index)
    return se3


# @jaxtyped(typechecker=beartype)
def nn(index: Int64[Tensor, "*batch"]) -> Int64[Tensor, "*batch 12"]:
    """
    Get nearest neighbour cell indices for the given index.

    Args:
        index: Cell index

    Returns:
        Nearest neighbour cell indices

    Notes:
        For boundary cells, we return the index of the cell itself for the out-of-bounds neighbours.
    """
    r3_index, so3_index = split_index(index)

    nn_r3, nn_so3 = split_index(index.unsqueeze(-1).repeat(*([1] * index.ndim), 12))
    nn_r3[..., :6] = r3.nn(r3_index)
    nn_so3[..., 6:] = so3.nn(so3_index)

    nn = combine_index(nn_r3, nn_so3)
    return nn


# @jaxtyped(typechecker=beartype)
def random(num_samples: int) -> Float[Tensor, "num_samples 4 4"]:
    """
    Sample random poses uniformly from SE(3).

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Random poses.
    """
    pose = torch.eye(4).repeat(num_samples, 1, 1)
    pose[:, :3, :3] = so3.random(num_samples)
    pose[:, :3, 3] = r3.random(num_samples)
    return pose


# @jaxtyped(typechecker=beartype)
def random_ball(num_samples: int,
                centre: Float[Tensor, "3"],
                radius: float) -> Float[Tensor, "num_samples 4 4"]:
    """
    Sample random poses uniformly from a bounding ball.

    Args:
        num_samples: Number of samples to generate.
        centre: Ball centre.
        radius: Ball radius.

    Returns:
        Random poses.
    """
    pose = torch.eye(4).repeat(num_samples, 1, 1)
    pose[:, :3, :3] = so3.random(num_samples)
    pose[:, :3, 3] = r3.random_ball(num_samples, centre, radius)
    return pose.to(centre.device)


# @jaxtyped(typechecker=beartype)
def to_vector(pose: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 9"]:
    """
    Convert 4x4 pose represented by a homogeneous transformation matrix to a 9D vector representation.

    Args:
        pose: Homogeneous transformation matrix

    Returns:
        9D vector representation
    """
    return torch.cat([pose[..., :3, 3], so3.to_vector(pose[..., :3, :3])], dim=-1)


# @jaxtyped(typechecker=beartype)
def from_vector(vec: Float[Tensor, "*batch 9"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Convert 9D vector representation to 4x4 homogeneous transformation matrix

    Args:
        vec: 9D vector representation

    Returns:
        Homogeneous transformation matrix
    """
    translation = vec[..., :3]
    rotation_cont = vec[..., 3:]
    batch_shape = vec.shape[:-1]
    homogeneous = torch.eye(4, device=vec.device).expand(*batch_shape, 4, 4).clone()
    homogeneous[..., :3, 3] = translation
    homogeneous[..., :3, :3] = so3.from_vector(rotation_cont)
    return homogeneous
