import torch
from jaxtyping import Float
from torch import Tensor


def quaternion_to_rotation_matrix(quaternions: Float[Tensor, "batch_size 4"]) -> Float[Tensor, "batch_size 3 3"]:
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Batched quaternions in (w, x, y, z) format
    Returns:
        Batched rotation matrices
    """
    w, x, y, z = quaternions.unbind(-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation_matrix = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)
    return rotation_matrix

def rotation_matrix_to_axis_angle(rotation_matrix: Float[Tensor, "batch_size 3 3"], epsilon: float = 1e-6) \
        -> Float[Tensor, "batch_size 3"]:
    """
    Convert rotation matrix to axis-angle vector.

    Args:
        rotation_matrix: Rotation matrix
        epsilon: Values smaller than epsilon are considered zero

    Returns:
        Axis-angle vector
    """
    rotation_matrix[torch.abs(rotation_matrix) < epsilon] = 0

    trace = torch.vmap(torch.trace)(rotation_matrix)
    angle = torch.acos((trace - 1) / 2)

    # if trace->3, acos->0. There are numerical issues causing nan value
    if torch.any((torch.abs(trace - 3) < epsilon)):
        angle[torch.abs(trace - 3) < epsilon] = 0
    # if trace->-1, acos->-pi.
    if torch.any((torch.abs(trace - (-1)) < epsilon)):
        angle[torch.abs(trace - (-1)) < epsilon] = torch.pi

    # Avoid division by zero by setting a small epsilon for angles near zero
    sin_angle = torch.sin(angle).clamp(min=epsilon)

    # Compute the rotation axis for each matrix
    # By definition, the rotation axis is the eigenvector corresponding to the eigenvalue 1
    axis_1 = torch.stack([
        (rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]) / (2 * sin_angle),
        (rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]) / (2 * sin_angle),
        (rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]) / (2 * sin_angle)
    ], dim=-1)

    # The above computation does not work when R is symmetric, so we compute axis_2 in this case.
    I = torch.eye(3, device=rotation_matrix.device)
    for _ in range(len(rotation_matrix.shape[:-2])):
        I = I.unsqueeze(0)
    D, P = torch.linalg.eig(rotation_matrix - I)


    vector_idx = torch.argmax((torch.abs(D.real) < epsilon).to(torch.int8), dim=-1, keepdim=True)
    axis_2 = torch.gather(P.real, -1, index=vector_idx.repeat_interleave(repeats=3, dim=-1)[..., None]).squeeze()
    axis = torch.where(
        torch.logical_or(
            torch.all((rotation_matrix.transpose(-2, -1) - rotation_matrix) < epsilon, dim=(-2, -1)),
            torch.isclose(angle, torch.tensor([torch.pi], device=angle.device))
        )[..., None], axis_2, axis_1)

    # Handle cases where the angle is close to zero by setting axis to zero
    axis = torch.where(angle[..., None] < epsilon, torch.zeros_like(axis), axis)
    if torch.any(axis.isnan()):
        raise ValueError('nan axis')

    # Return the axis-angle vector (axis * angle)
    return axis * angle[..., None]