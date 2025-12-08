import math
import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int

import data_sampling.r3 as r3
import data_sampling.so3 as so3
from data_sampling.orientation_representations import rotation_matrix_to_cont
from scipy.spatial.transform import Rotation

#@jaxtyped(typechecker=beartype)
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


#@jaxtyped(typechecker=beartype)
def _split_index(index: Int[Tensor, "*batch"]) -> tuple[Int[Tensor, "*batch"], Int[Tensor, "*batch"]]:
    return index % r3.N_CELLS, index // r3.N_CELLS


#@jaxtyped(typechecker=beartype)
def _combine_index(r3_index: Int[Tensor, "*batch"], so3_index: Int[Tensor, "*batch"]) -> Int[Tensor, "*batch"]:
    return r3_index + so3_index * r3.N_CELLS


#@jaxtyped(typechecker=beartype)
def index(pose: Float[Tensor, "*batch 4 4"]) -> Int[Tensor, "*batch"]:
    """
    Get cell index for the given poses.

    Args:
        pose: Pose.
    Returns:
        Cell index.
    """
    return _combine_index(r3.index(pose[:, :3, 3]), so3.index(pose[:, :3, :3]))


#@jaxtyped(typechecker=beartype)
def cell(index: Int[Tensor, "*batch"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Get cell pose for the given index.

    Args:
        index: Cell index.

    Returns:
        Cell pose.
    """
    r3_index, so3_index = _split_index(index)
    se3 = torch.eye(4).repeat(*index.shape, 1, 1)
    se3[..., :3, 3] = r3.cell(r3_index)
    se3[..., :3, :3] = so3.cell(so3_index)
    return se3


#@jaxtyped(typechecker=beartype)
def cell_vec(index: Int[Tensor, "*batch"]) -> Float[Tensor, "*batch 9"]:
    """
    Get cell pose for the given index.

    Args:
        index: Cell index.

    Returns:
        Cell pose vectorized.
    """
    r3_index, so3_index = _split_index(index)
    return torch.cat([r3.cell(r3_index), rotation_matrix_to_cont(so3.cell(so3_index))], dim=-1)


#@jaxtyped(typechecker=beartype)
def nn(index: Int[Tensor, "*batch"]) -> Int[Tensor, "*batch 12"]:
    """
    Get nearest neighbour cell indices for the given index.

    Args:
        index: Cell index

    Returns:
        Nearest neighbour cell indices

    Notes:
        For boundary cells, we return the index of the cell itself for the out-of-bounds neighbours.
    """
    r3_index, so3_index = _split_index(index)

    nn_r3, nn_so3 = _split_index(index.unsqueeze(-1).repeat(*([1] * index.ndim), 12))
    nn_r3[..., :6] = r3.nn(r3_index)
    nn_so3[..., 6:] = so3.nn(so3_index)

    nn = _combine_index(nn_r3, nn_so3)
    return nn


#@jaxtyped(typechecker=beartype)
def random(num_samples: int) -> Float[Tensor, "num_samples 4 4"]:
    """
    Sample random poses uniformly from SE(3).

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Random poses.
    """
    translation = torch.randn(num_samples, 3)
    translation /= torch.norm(translation, dim=1, keepdim=True)
    translation *= torch.pow(torch.rand(num_samples, 1), 1.0 / 3)

    quaternion = torch.randn(num_samples, 4)
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
    rotation = Rotation.from_quat(quaternion).as_matrix()

    pose = torch.eye(4).repeat(num_samples, 1, 1)
    pose[:, :3, :3] = rotation
    pose[:, :3, 3] = translation
    return pose