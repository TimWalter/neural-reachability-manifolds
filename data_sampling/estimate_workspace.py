from typing import Union

import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int

from autotune_batch_size import get_batch_size
from data_sampling.robotics import LINK_RADIUS, forward_kinematics, geometric_jacobian, yoshikawa_manipulability, \
    collision_check
from data_sampling.se3_cells import SE3_CELLS, N_CELLS, R3_LOOKUP_MIN_DISTANCE, SO3_LOOKUP_MIN_DISTANCE, se3_indices


@jaxtyped(typechecker=beartype)
def unique_poses(poses: Float[Tensor, "batch 4 4"],
                 manip_idx: Float[Tensor, "batch"],
                 joints: Float[Tensor, "batch dofp1"],
                 cell_indices: Int[Tensor, "batch"]) \
        -> tuple[
            Float[Tensor, "n_unique 4 4"],
            Float[Tensor, "n_unique"],
            Float[Tensor, "n_unique dofp1"],
            Int[Tensor, "n_unique"]]:
    """
    Select unique poses based on cell indices, keeping the ones with the highest manipulability index.

    Args:
        poses: Poses in SE(3)
        manip_idx: Manipulability indices
        joints: Joint configurations corresponding to poses
        cell_indices: Cell indices corresponding to poses

    Returns:
        Unique poses, their manipulability indices, and their cell indices
    """
    manip_idx, sort_indices = torch.sort(manip_idx, descending=True)
    cell_indices = cell_indices[sort_indices]
    poses = poses[sort_indices]
    joints = joints[sort_indices]

    cell_indices, inverse, counts = torch.unique(cell_indices, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    unique_indices = inv_sorted[tot_counts]

    poses = poses[unique_indices]
    manip_idx = manip_idx[unique_indices]
    joints = joints[unique_indices]

    return poses, manip_idx, joints, cell_indices


@jaxtyped(typechecker=beartype)
def sample_joints(mdh: Float[Tensor, "dofp1 3"], batch_size: int,
                  inp_joints: Float[Tensor, "batch_size dofp1"],
                  inp_manip) \
        -> Float[Tensor, "{batch_size} dofp1"]:
    """
    Sample random joint configurations for the robot.

    Args:
        mdh: MDH parameters encoding the robot geometry
        batch_size: Number of joint configurations to sample

    Returns:
        Joint configurations
    """
    # TODO rethink how we sample joints, also where have we been before?
    joints = 2 * torch.pi * torch.rand(batch_size, mdh.shape[0] - 1, device=mdh.device) - torch.pi
    valid_joints = inp_joints[:, :-1][inp_manip != -1]
    valid_manip_idx = inp_manip[inp_manip != -1]
    n_exploit = min(valid_joints.shape[0], batch_size)
    if n_exploit != 0:
        idx = torch.randint(0, valid_joints.shape[0], (n_exploit,))

        base_manip = valid_manip_idx[idx]
        base_manip = (base_manip - base_manip.min()) / (base_manip.max() - base_manip.min())

        exploited = valid_joints[idx] + torch.pi * 0.1 - base_manip.unsqueeze(1) * (
                torch.pi * 0.1 - 0.01) * torch.randn_like(valid_joints[idx])
        exploited = (exploited + torch.pi) % (2 * torch.pi) - torch.pi
        joints[:n_exploit] = exploited.to(mdh.device)

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
    poses = SE3_CELLS
    # Ensure diverse unreachable poses
    poses[:, :3, 3] += torch.clamp(torch.randn_like(poses[:, :3, 3]) / 3, -1, 1) * R3_LOOKUP_MIN_DISTANCE
    poses[:, :3, :3] += torch.clamp(torch.randn_like(poses[:, :3, :3]) / 3, -1, 1) * SO3_LOOKUP_MIN_DISTANCE

    manip_idx = -torch.ones(N_CELLS, device="cpu")
    joints = torch.zeros(N_CELLS, mdh.shape[0], device="cpu")

    def workload(batch_size: int, mdh: Float[Tensor, "dof 3"], link_radius: float) \
            -> tuple[
                Float[Tensor, "batch_size 4 4"],
                Float[Tensor, "batch_size"],
                Float[Tensor, "batch_size"],
                Float[Tensor, "batch_size dof"]]:
        batch_joints = sample_joints(mdh, batch_size, joints, manip_idx)
        batch_mdh = mdh.unsqueeze(0).expand(batch_size, -1, -1)
        batch_poses = forward_kinematics(batch_mdh, batch_joints.unsqueeze(-1))

        self_collision = collision_check(batch_mdh, batch_poses, link_radius)
        batch_poses = batch_poses[~self_collision]
        batch_joints = batch_joints[~self_collision]
        jacobian = geometric_jacobian(batch_poses)
        batch_manip_idx = yoshikawa_manipulability(jacobian)
        batch_poses = batch_poses[:, -1, :, :]
        batch_cell_indices = se3_indices(batch_poses)
        batch_poses, batch_manip_idx, batch_joints, batch_cell_indices = unique_poses(batch_poses, batch_manip_idx,
                                                                                      batch_joints,
                                                                                      batch_cell_indices)

        return batch_poses.cpu(), batch_manip_idx.cpu(), batch_cell_indices.cpu(), batch_joints.cpu()

    args = (mdh, LINK_RADIUS)
    batch_size = get_batch_size(mdh.device, workload, args)

    threshold = int(min(batch_size, N_CELLS) * 1e-3)
    newly_filled_cells = threshold + 1  # just to get the loop going
    while newly_filled_cells > threshold:
        batch_poses, batch_manip_idx, batch_cell_indices, batch_joints = workload(batch_size, *args)

        prev = manip_idx[batch_cell_indices]
        update_mask = batch_manip_idx > prev
        newly_filled_cells = (manip_idx[batch_cell_indices[update_mask]] == -1).sum().item()
        poses[batch_cell_indices[update_mask]] = batch_poses[update_mask]
        manip_idx[batch_cell_indices[update_mask]] = batch_manip_idx[update_mask]
        joints[batch_cell_indices[update_mask]] = batch_joints[update_mask]

    if not full_poses:
        poses = torch.cat([poses[:, :3, 3], poses[:, :3, :2].reshape(-1, 6)], dim=-1)

    return poses, manip_idx, joints


if __name__ == "__main__":
    import pyarrow.parquet
    from pathlib import Path

    table = pyarrow.parquet.read_table(Path(__file__).parent.parent / "data" / "train" / "3_0.parquet")
    data = {
        "mdh": torch.from_numpy(table['mdh'].combine_chunks().to_numpy_ndarray()),
        "poses": torch.from_numpy(table['poses'].combine_chunks().to_numpy_ndarray()),
        "labels": torch.from_numpy(table['labels'].to_numpy(zero_copy_only=False)),
        "joints": torch.from_numpy(table['joints'].combine_chunks().to_numpy_ndarray()),
    }
    idx = torch.where(data["labels"] != -1)[0][0]
    mdh = data["mdh"][idx].view(-1, 3)[:4, :].to("cuda")

    poses, manip_idx, joints = estimate_workspace(mdh, full_poses=True)

    print("pause")
