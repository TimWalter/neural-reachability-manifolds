import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from scipy.spatial.transform import Rotation


def homogeneous_to_vector(homogeneous: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 9"]:
    """
    Convert 4x4 homogeneous transformation matrix to 9D vector representation

    Args:
        homogeneous: Homogeneous transformation matrix

    Returns:
        9D vector representation
    """
    return torch.cat([homogeneous[..., :3, 3], rotation_matrix_to_continuous(homogeneous[..., :3, :3])], dim=-1)


def vector_to_homogeneous(vec: Float[Tensor, "*batch 9"]) -> Float[Tensor, "*batch 4 4"]:
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
    homogeneous[..., :3, :3] = continuous_to_rotation_matrix(rotation_cont)
    return homogeneous


# @jaxtyped(typechecker=beartype)
def rotation_matrix_to_continuous(rotation_matrix: Float[Tensor, "*batch 3 3"]) -> Float[Tensor, "*batch 6"]:
    """
    Convert 3x3 rotation matrix to a continuous 6D rotation representation

    Args:
        rotation_matrix: Rotation matrix

    Returns:
        6D rotation representation
    """
    return rotation_matrix[..., :3, :2].transpose(-1, -2).reshape(*rotation_matrix.shape[:-2], 6)


# @jaxtyped(typechecker=beartype)
def continuous_to_rotation_matrix(ml: Float[Tensor, "*batch 6"]) -> Float[Tensor, "*batch 3 3"]:
    """
    Convert continuous 6D rotation representation to 3x3 rotation matrix.

    Args:
        ml: 6D rotation representation

    Returns:
        Rotation matrix
    """
    r1 = ml[..., :3]
    r2 = ml[..., 3:]
    r3 = torch.cross(r1, r2, dim=-1)
    return torch.stack([r1, r2, r3], dim=-1)


@torch.compile
def rotation_matrix_to_rotation_vector(rotation_matrix: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 3"]:
    """
    Convert 3x3 rotation matrix to rotation vector (axis-angle representation).

    Args:
        rotation_matrix: Rotation matrix

    Returns:
        Rotation vector
    """
    return Rotation.from_matrix(rotation_matrix, assume_valid=True).as_rotvec()
