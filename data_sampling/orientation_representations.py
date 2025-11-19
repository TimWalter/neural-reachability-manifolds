import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from scipy.spatial.transform import Rotation


@jaxtyped(typechecker=beartype)
def rotation_matrix_to_ml(rotation_matrix: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 6"]:
    """
    Convert 3x3 rotation matrix to 6D rotation representation

    Args:
        rotation_matrix: Rotation matrix

    Returns:
        6D rotation representation
    """
    return rotation_matrix[..., :3, :2].reshape(-1, 6)

@jaxtyped(typechecker=beartype)
def ml_to_rotation_matrix(ml: Float[Tensor, "*batch 6"]) -> Float[Tensor, "*batch 3 3"]:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        ml: 6D rotation representation

    Returns:
        Rotation matrix
    """
    r1 = ml[..., :3]
    r2 = ml[..., 3:]
    r3 = torch.cross(r1, r2, dim=-1)
    return torch.stack([r1, r2, r3], dim=-1)


@jaxtyped(typechecker=beartype)
def rotation_matrix_to_rotation_vector(rotation_matrix: Float[Tensor, "batch 3 3"]) \
        -> Float[Tensor, "batch 3"]:
    """
    Convert rotation matrix to rotation vector (axis-angle).

    Args:
        rotation_matrix: Rotation matrix
        epsilon: Values smaller than epsilon are considered zero

    Returns:
        Axis-angle vector
    """
    return torch.from_numpy(Rotation.from_matrix(rotation_matrix.cpu()).as_rotvec()).to(
        device=rotation_matrix.device, dtype=rotation_matrix.dtype)


@jaxtyped(typechecker=beartype)
def rotation_vector_to_quaternion(rotation_vectors: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch 4"]:
    """
    Convert rotation vectors (axis-angle) to quaternions.

    Args:
        rotation_vectors: Batched rotation vectors

    Returns:
        Batched quaternions in (w, x, y, z) format
    """
    return torch.from_numpy(Rotation.from_rotvec(rotation_vectors.cpu()).as_quat()).float().to(
        device=rotation_vectors.device, dtype=rotation_vectors.dtype)


@jaxtyped(typechecker=beartype)
def quaternion_to_rotation_matrix(quaternions: Float[Tensor, "batch 4"]) -> Float[Tensor, "batch 3 3"]:
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Batched quaternions in (a, b, c, d) format
    Returns:
        Batched rotation matrices
    """
    return torch.from_numpy(Rotation.from_quat(quaternions.cpu()).as_matrix()).float().to(
        device=quaternions.device, dtype=quaternions.dtype)


@jaxtyped(typechecker=beartype)
def rotation_matrix_to_quaternion(rotation_matrix: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 4"]:
    """
    Convert rotation matrices to quaternions.

    Args:
        rotation_matrix: Rotation matrix
    Returns:
        Batched quaternions
    """
    return torch.from_numpy(Rotation.from_matrix(rotation_matrix.cpu()).as_quat()).float().to(
        device=rotation_matrix.device, dtype=rotation_matrix.dtype)


@jaxtyped(typechecker=beartype)
def rotation_vector_to_rotation_matrix(rotation_vectors: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch 3 3"]:
    return torch.from_numpy(Rotation.from_rotvec(rotation_vectors.cpu()).as_matrix()).to(
        device=rotation_vectors.device, dtype=rotation_vectors.dtype)