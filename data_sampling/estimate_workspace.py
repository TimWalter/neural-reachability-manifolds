import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int

from data_sampling.autotune_batch_size import get_batch_size
from data_sampling.robotics import forward_kinematics, geometric_jacobian, yoshikawa_manipulability, \
    collision_check
from data_sampling.se3_cells import N_CELLS, se3_indices


@jaxtyped(typechecker=beartype)
def unique_cells(cell_indices: Int[Tensor, "batch"],
                 labels: Float[Tensor, "batch"],
                 joints: Float[Tensor, "batch dofp1"]) \
        -> tuple[
            Int[Tensor, "n_unique"],
            Float[Tensor, "n_unique"],
            Float[Tensor, "n_unique dofp1"]
        ]:
    """
    Select unique cell indices, keeping the ones with the highest label.

    Args:
        cell_indices: Cell indices corresponding to poses
        labels: Labels to sort by
        joints: Joint configurations corresponding to poses

    Returns:
        Unique cell indices with their respective labels and joint configurations
    """
    labels, sort_indices = torch.sort(labels, descending=True)
    cell_indices = cell_indices[sort_indices]
    joints = joints[sort_indices]

    cell_indices, inverse, counts = torch.unique(cell_indices, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    unique_indices = inv_sorted[tot_counts]

    labels = labels[unique_indices]
    joints = joints[unique_indices]

    return cell_indices, labels, joints


@jaxtyped(typechecker=beartype)
def sample_joints(batch_size: int, joints: Float[Tensor, "batch dofp1"], labels: Float[Tensor, "N_CELLS"]) \
        -> Float[Tensor, "{batch_size} dofp1"]:
    """
    Sample random joint configurations for the robot.

    Args:
        batch_size: Number of joint configurations to sample
        joints: Previous joint configurations
        labels: Labels encoding the capability map

    Returns:
        Newly sampled joint configurations
    """
    mask = labels != -1
    joints = joints[mask][torch.argsort(labels[mask], descending=True)]

    joints = joints[:batch_size] + torch.randn_like(joints[:batch_size])
    joints = torch.atan2(torch.sin(joints), torch.cos(joints))
    joints[:, -1] = 0.0
    missing_samples = batch_size - joints.shape[0]
    joints = torch.cat([
        joints,
        torch.cat(
            [2 * torch.pi * torch.rand(missing_samples, joints.shape[1] - 1) - torch.pi,
             torch.zeros(missing_samples, 1)],
            dim=1)
    ], dim=0)

    return joints


def workload(batch_size: int,
             morph: Float[Tensor, "dof 3"],
             labels: Float[Tensor, "N_CELLS"],
             joints: Float[Tensor, "N_CELLS dofp1"]) -> tuple[
    Float[Tensor, "batch_size"],
    Float[Tensor, "batch_size"],
    Float[Tensor, "batch_size dof"]]:
    """
    Computes one batch of the capability map estimation.
    Args:
        batch_size: Number of samples to process in this batch
        morph: MDH parameters encoding the robot geometry
        labels: Labels encoding the capability map
        joints: Joint configurations corresponding to labels

    Returns:
        Labels, cell indices, and joint configurations for the batch
    """
    morph = morph.unsqueeze(0).expand(batch_size, -1, -1)

    new_joints = sample_joints(batch_size, joints, labels).to(morph.device)
    new_poses = forward_kinematics(morph, new_joints.unsqueeze(-1))

    self_collision = collision_check(morph, new_poses)
    new_poses = new_poses[~self_collision]
    new_joints = new_joints[~self_collision]

    jacobian = geometric_jacobian(new_poses)
    new_labels = yoshikawa_manipulability(jacobian)

    new_cell_indices = se3_indices(new_poses[:, -1, :, :])

    new_cell_indices, new_labels, new_joints = unique_cells(new_cell_indices, new_labels, new_joints)

    return new_labels.cpu(), new_cell_indices.cpu(), new_joints.cpu()


@jaxtyped(typechecker=beartype)
def estimate_workspace(morph: Float[Tensor, "dofp1 3"]) -> Float[Tensor, "n_cells"]:
    """
    Sample the robot's workspace and return the estimated capability map.

    Args:
        morph: MDH parameters encoding the robot geometry

    Returns:
        Labels encoding the discretized capability map
    """
    morph = morph.to("cuda")

    labels = -torch.ones(N_CELLS, device="cpu")
    joints = torch.zeros(N_CELLS, morph.shape[0], device="cpu")

    args = (morph, labels, joints)
    batch_size = get_batch_size(morph.device, workload, args)

    threshold = int(min(batch_size, N_CELLS) * 1e-3)
    newly_filled_cells = threshold + 1  # just to get the loop going
    while newly_filled_cells > threshold:
        new_labels, new_cell_indices, new_joints = workload(batch_size, *args)

        old_labels = labels[new_cell_indices]
        new_update_mask = new_labels > old_labels
        old_update_indices = new_cell_indices[new_update_mask]

        labels[old_update_indices] = new_labels[new_update_mask]
        joints[old_update_indices] = new_joints[new_update_mask]

        newly_filled_cells = torch.sum(old_labels == -1).item()

    return labels
