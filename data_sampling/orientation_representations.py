import jax
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from scipy.spatial.transform._rotation_xp import _from_matrix_orthogonal, as_rotvec


# @jaxtyped(typechecker=beartype)
def rotation_matrix_to_cont(rotation_matrix: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 6"]:
    """
    Convert 3x3 rotation matrix to a continuous 6D rotation representation

    Args:
        rotation_matrix: Rotation matrix

    Returns:
        6D rotation representation
    """
    return rotation_matrix[..., :3, :2].reshape(-1, 6)


# @jaxtyped(typechecker=beartype)
def cont_to_rotation_matrix(ml: Float[Tensor, "*batch 6"]) -> Float[Tensor, "*batch 3 3"]:
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


# Jax is almost an order of magnitude faster than torch for this operation
@jax.jit
def rotation_matrix_to_rotation_vector_jax(rotation):
    return as_rotvec(_from_matrix_orthogonal(rotation))


@torch._dynamo.disable
def rotation_matrix_to_rotation_vector(rotation_matrix: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 3"]:
    """
    Convert 3x3 rotation matrix to rotation vector (axis-angle representation).

    Args:
        rotation_matrix: Rotation matrix

    Returns:
        Rotation vector
    """
    rotation_vector = rotation_matrix_to_rotation_vector_jax(jax.dlpack.from_dlpack(rotation_matrix.contiguous()))
    return torch.from_dlpack(rotation_vector)
