from typing import Union

import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped

from data_sampling.autotune_batch_size import get_batch_size
from data_sampling.robotics import LINK_RADIUS, forward_kinematics, geometric_jacobian, yoshikawa_manipulability, \
    collision_check
from data_sampling.se3_cells import SE3_CELLS, N_CELLS, R3_LOOKUP_MIN_DISTANCE, SO3_LOOKUP_MIN_DISTANCE, se3_indices

NUM_SAMPLES = 2_000_000


@jaxtyped(typechecker=beartype)
def sample_joints(mdh: Float[Tensor, "dofp1 3"], batch_size: int) -> Float[Tensor, "{batch_size} dofp1"]:
    """
    Sample random joint configurations for the robot.

    Args:
        mdh: MDH parameters encoding the robot geometry
        batch_size: Number of joint configurations to sample

    Returns:
        Joint configurations
    """
    joints = 2 * torch.pi * torch.rand(batch_size, mdh.shape[0] - 1, device=mdh.device) - torch.pi
    return torch.cat([joints, torch.zeros(batch_size, 1).to(mdh.device)], dim=1)


@jaxtyped(typechecker=beartype)
def estimate_workspace(mdh: Float[Tensor, "dofp1 3"], full_poses: bool = False) \
        -> tuple[
            Union[
                Float[Tensor, "n_cells 4 4"],
                Float[Tensor, "n_cells 9"]
            ],
            Float[Tensor, "n_cells"],
            Float[Tensor, "n_cells dofp1"]
        ]:
    """
    Sample the robot's workspace and return poses and manipulability indices.

    Args:
        mdh: MDH parameters encoding the robot geometry
        full_poses: Whether to return full 4x4 poses or a continuous 6D representation

    Returns:
        Sampled poses and corresponding manipulability indices and joints
    """
    mdh = mdh.to("cuda")
    unreachable_indices = torch.arange(N_CELLS, device="cpu")

    poses = torch.empty(0, 4, 4, device="cpu")
    manip_idx = torch.empty(0, device="cpu")
    joints = torch.empty(0, mdh.shape[0], device="cpu")

    def workload(batch_size: int, mdh: Float[Tensor, "dof 3"], link_radius: float) \
            -> tuple[
                Float[Tensor, "batch_size 4 4"],
                Float[Tensor, "batch_size"],
                Float[Tensor, "batch_size"],
                Float[Tensor, "batch_size dof"]]:
        batch_joints = sample_joints(mdh, batch_size)
        batch_mdh = mdh.unsqueeze(0).expand(batch_size, -1, -1)
        batch_poses = forward_kinematics(batch_mdh, batch_joints.unsqueeze(-1))

        self_collision = collision_check(batch_mdh, batch_poses, link_radius)
        batch_poses = batch_poses[~self_collision]
        batch_joints = batch_joints[~self_collision]

        jacobian = geometric_jacobian(batch_poses)
        batch_manip_idx = yoshikawa_manipulability(jacobian)

        batch_poses = batch_poses[:, -1, :, :]
        batch_cell_indices = se3_indices(batch_poses)

        return batch_poses.cpu(), batch_manip_idx.cpu(), batch_cell_indices.cpu(), batch_joints.cpu()

    args = (mdh, LINK_RADIUS)
    batch_size = get_batch_size(mdh.device, workload, args)

    threshold = int(min(batch_size, N_CELLS) * 1e-3)
    newly_filled_cells = threshold + 1  # just to get the loop going
    timeout = 0
    while (newly_filled_cells > threshold or manip_idx.shape[0] < NUM_SAMPLES / 2) and timeout < 1000:
        batch_poses, batch_manip_idx, batch_cell_indices, batch_joints = workload(batch_size, *args)

        poses = torch.cat([poses, batch_poses], dim=0)
        manip_idx = torch.cat([manip_idx, batch_manip_idx], dim=0)
        joints = torch.cat([joints, batch_joints], dim=0)

        mask = ~torch.isin(unreachable_indices, batch_cell_indices)
        unreachable_indices = unreachable_indices[mask]
        newly_filled_cells =mask.shape[0] - mask.sum().item()

    unreachable_indices = unreachable_indices[torch.randperm(unreachable_indices.shape[0])[:NUM_SAMPLES // 2]]
    unreachable_poses = SE3_CELLS[unreachable_indices]
    # Ensure diverse unreachable poses
    unreachable_poses[:, :3, 3] += torch.clamp(torch.randn_like(unreachable_poses[:, :3, 3]) / 3, -1,
                                               1) * R3_LOOKUP_MIN_DISTANCE
    unreachable_poses[:, :3, :3] += torch.clamp(torch.randn_like(unreachable_poses[:, :3, :3]) / 3, -1,
                                                1) * SO3_LOOKUP_MIN_DISTANCE
    unreachable_samples = unreachable_indices.shape[0]

    poses = torch.cat([poses[:NUM_SAMPLES - unreachable_samples], unreachable_poses], dim=0)
    manip_idx = torch.cat([manip_idx[:NUM_SAMPLES - unreachable_samples], -torch.ones(unreachable_samples)], dim=0)
    joints = torch.cat([joints[:NUM_SAMPLES - unreachable_samples], torch.zeros(unreachable_samples, mdh.shape[0])],
                       dim=0)

    if not full_poses:
        poses = torch.cat([poses[:, :3, 3], poses[:, :3, :2].reshape(-1, 6)], dim=-1)

    return poses, manip_idx, joints
