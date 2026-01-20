import torch

from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool


# @jaxtyped(typechecker=beartype)
def geometric_jacobian(poses: Float[Tensor, "*batch dofp1 4 4"]) -> Float[Tensor, "*batch 6 dof"]:
    """
    Compute the geometric Jacobian for revolute joints in the base frame.

    Args:
        poses: Homogeneous transforms (batch, dof, 4, 4) from base to each joint, including the end-effector
               as the last transform.

    Returns:
        Jacobian where the first 3 rows are linear velocity components and the last 3 rows are angular velocity components.
    """
    orientation = poses[..., :-1, :3, :3]
    positions = poses[..., :-1, :3, 3]
    eef_position = poses[..., -1:, :3, 3]
    joint_z_axes = orientation[..., :, 2]
    jacobian = torch.cat([torch.cross(joint_z_axes, eef_position - positions, dim=-1), joint_z_axes],
                         dim=-1).transpose(-1, -2)

    return jacobian


# @jaxtyped(typechecker=beartype)
def yoshikawa_manipulability(jacobian: Float[Tensor, "*batch 6 dof"], soft: bool = False) -> Float[Tensor, "*batch"]:
    """
    Computes the Yoshikawa manipulability index using the SVD of the Jacobian.

    Args:
        jacobian: Geometric Jacobians for each robot configuration.
        soft: If true, assume the workspace dimension is equal to dof.

    Returns:
        Manipulability index.
    """
    if soft and jacobian.shape[-1] < 6:
        _, singular_values, _ = torch.linalg.svd(jacobian)
        manipulability = torch.prod(singular_values[..., : min(6, jacobian.shape[-1])], dim=-1)
    else:
        manipulability = torch.sqrt(torch.det(torch.matmul(jacobian, jacobian.transpose(-1, -2))).abs())
    return manipulability
