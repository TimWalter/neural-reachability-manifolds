import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int, Bool

from data_sampling.autotune_batch_size import get_batch_size
from data_sampling.robotics import forward_kinematics, collision_check, analytical_inverse_kinematics

import data_sampling.se3 as se3
import data_sampling.r3 as r3
import data_sampling.so3 as so3

torch.set_float32_matmul_precision("high")


# @jaxtyped(typechecker=beartype)
def sample_joints(batch_size: int, dof: int) -> Float[Tensor, "{batch_size} {dof+1}"]:
    """
    sample random joint configurations for the robot.

    Args:
        batch_size: Number of joint configurations to sample
        dof: Degrees of freedom of the robot + 1

    Returns:
        Newly sampled joint configurations
    """
    joints = torch.cat(
        [
            2 * torch.pi * torch.rand(batch_size, dof - 1, device="cuda") - torch.pi,
            torch.zeros(batch_size, 1, device="cuda"),
        ],
        dim=1,
    )

    return joints


# @jaxtyped(typechecker=beartype)
def sample_reachable_eef_poses(batch_size: int, morph: Float[Tensor, "dof 3"]) -> Float[Tensor, "n_valid 4 4"]:
    """
    Sample end effector poses for the robot.

    Args:
        batch_size: Number of end effector poses to sample.
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Reachable end effector poses.
    """
    joints = sample_joints(batch_size, morph.shape[0])
    reached_poses = forward_kinematics(morph.unsqueeze(0).expand(batch_size, -1, -1), joints.unsqueeze(-1))
    self_collision = collision_check(morph.unsqueeze(0).expand(reached_poses.shape[0], -1, -1), reached_poses)

    return reached_poses[:, -1, :, :][~self_collision]


@jaxtyped(typechecker=beartype)
def estimate_workspace_analytically(morph: Float[Tensor, "dof 3"], num_samples: int) -> tuple[
    Float[Tensor, " {num_samples} 4 4"],
    Bool[Tensor, " {num_samples}"]
]:
    """
    Estimate the workspace of a robot with analytically solvable inverse kinematics.
    Estimation by three parts:
        1. 50% Reachable samples via FK
        2. 1% Unreachable samples uniformly sampled in the workspace.
        3. 49% Unreachable samples within the reachable R3 region approximated by a ball.

    Args:
        morph: MDH parameters encoding the robot geometry
        num_samples: Number of samples to generate
    Returns:
        Poses and labels encoding the discretised capability map
    """
    assert num_samples % 100 == 0

    morph = morph.to("cuda")
    poses = torch.eye(4).repeat(num_samples, 1, 1)
    labels = torch.cat(
        [torch.ones(num_samples // 2, dtype=torch.bool), torch.zeros(num_samples // 2, dtype=torch.bool)]
    )

    idx = 0
    oversampling = 2
    while idx != num_samples * 50 // 100:
        remaining_samples = num_samples * 50 // 100 - idx

        candidate_poses = sample_reachable_eef_poses(oversampling * remaining_samples, morph).cpu()
        candidate_poses = candidate_poses[:remaining_samples]

        poses[idx: idx + candidate_poses.shape[0]] = candidate_poses
        idx += candidate_poses.shape[0]
    morph = morph.to("cpu")
    reachable_center = torch.mean(poses[:idx, :3, 3], dim=0)
    reachable_radius = torch.norm(poses[:idx, :3, 3] - reachable_center, dim=1).max()

    while idx != num_samples * 51 // 100:
        remaining_samples = num_samples * 51 // 100 - idx

        candidate_poses = se3.random(oversampling * remaining_samples)
        joints, manipulability = analytical_inverse_kinematics(morph, candidate_poses)
        candidate_poses = candidate_poses[manipulability.cpu() == -1][:remaining_samples]

        poses[idx: idx + candidate_poses.shape[0]] = candidate_poses
        idx += candidate_poses.shape[0]

    oversampling = 5
    while idx != num_samples:
        remaining_samples = num_samples - idx
        candidates_samples = oversampling * remaining_samples

        candidate_poses = torch.eye(4).repeat(candidates_samples, 1, 1)
        candidate_poses[..., :3, :3] = so3.random(candidates_samples)
        radius = reachable_radius * torch.rand(candidates_samples) ** (1 / 3)
        direction = torch.randn(candidates_samples, 3)
        direction /= torch.norm(direction, dim=1, keepdim=True)
        candidate_poses[..., :3, 3] = reachable_center + radius.unsqueeze(1) * direction
        joints, manipulability = analytical_inverse_kinematics(morph, candidate_poses)
        candidate_poses = candidate_poses[manipulability.cpu() == -1][:remaining_samples]

        poses[idx: idx + candidate_poses.shape[0]] = candidate_poses
        idx += candidate_poses.shape[0]

    return poses, labels


# @jaxtyped(typechecker=beartype)
def sample_reachable_eef_cells(batch_size: int, morph: Float[Tensor, "dof 3"]) -> Int[Tensor, " n_cells"]:
    """
    Sample end effector cell indices for the robot.

    Args:
        batch_size: Number of samples to process in this batch
        morph: MDH parameters encoding the robot geometry

    Returns:
        Cell indices of reachable cells

    Notes:
        Cells with self-collisions are not necessarily unreachable, as the max operator is applied for a pose on all
        joints configurations
    """
    poses = sample_reachable_eef_poses(batch_size, morph)
    new_cell_indices = se3.index(poses)

    return new_cell_indices


compiled_sample_reachable_eef_cells = torch.compile(sample_reachable_eef_cells)


# @jaxtyped(typechecker=beartype)
def estimate_workspace(morph: Float[Tensor, "dofp1 3"]) -> tuple[Int[Tensor, " n_cells"], Bool[Tensor, " n_cells"]]:
    """
    Estimate the workspace of a robot solely from forward kinematics.
    Estimation by three parts:
        1. Reachable samples via FK
        2. Unreachable samples uniformly sampled in the workspace.
        3. Unreachable samples within the reachable R3 region approximated by a ball.

    Args:
        morph: MDH parameters encoding the robot geometry

    Returns:
        Labels and indices encoding the discretised capability map
    """
    morph = morph.to("cuda")

    # 1. Reachable samples via FK
    cell_indices = torch.empty(0, dtype=torch.int64, device="cuda")

    args = (morph,)
    batch_size = get_batch_size(morph.device, sample_reachable_eef_cells, args, safety=0.5)

    filled_cells = 0
    newly_filled_cells = 0
    while (newly_filled_cells == 0 and filled_cells == 0) or newly_filled_cells / filled_cells > 1e-2:
        new_cell_indices = compiled_sample_reachable_eef_cells(batch_size, *args)

        cell_indices = torch.cat([cell_indices, new_cell_indices]).unique()

        newly_filled_cells = len(cell_indices) - filled_cells
        filled_cells += newly_filled_cells

    cell_indices = cell_indices.cpu()

    # Dilate
    cell_indices = torch.cat([cell_indices, se3.nn(cell_indices).flatten()], dim=0).unique()

    labels = torch.ones(cell_indices.shape[0], dtype=torch.bool)

    # 2. Unreachable samples uniformly sampled in the workspace.
    u_indices = torch.randint(0, se3.N_CELLS, (2 * cell_indices.shape[0] // 50,)).unique()
    u_indices = u_indices[~torch.isin(u_indices, cell_indices)]
    u_indices = u_indices[:cell_indices.shape[0] // 50]

    # 3. Unreachable samples within the reachable R3 region approximated by a ball.
    positions = r3.cell(se3.split_index(cell_indices)[0])
    reachable_center = positions.mean(dim=0)
    reachable_radius = torch.norm(positions - reachable_center, dim=1).max()

    to_sample = min(cell_indices.shape[0], se3.N_CELLS - cell_indices.shape[0])
    if to_sample <= 0:
        print(morph)
        exit(1)

    cell_indices = torch.cat([cell_indices, u_indices], dim=0)
    labels = torch.cat([labels, torch.zeros_like(u_indices)], dim=0).bool()
    add_u_indices = torch.zeros(to_sample, dtype=torch.int64)
    oversampling = 5
    idx = 0
    while idx != to_sample:
        remaining_samples = to_sample - idx
        candidates_samples = oversampling * remaining_samples

        radius = reachable_radius * torch.rand(candidates_samples) ** (1 / 3)
        direction = torch.randn(candidates_samples, 3)
        direction /= torch.norm(direction, dim=1, keepdim=True)
        r3_index = r3.index(reachable_center + radius.unsqueeze(1) * direction)
        so3_index = torch.randint(0, so3.N_CELLS, (candidates_samples,))
        candidate_indices = se3.combine_index(r3_index, so3_index).unique()
        candidate_indices = candidate_indices[~torch.isin(candidate_indices, cell_indices)][:remaining_samples]

        add_u_indices[idx: idx + candidate_indices.shape[0]] = candidate_indices
        idx += candidate_indices.shape[0]

    cell_indices = torch.cat([cell_indices, add_u_indices], dim=0)
    labels = torch.cat([labels, torch.zeros_like(add_u_indices)], dim=0).bool()

    # Boundary as negative samples, TODO question: Do not do this and closed world assumption? do we want to oversample the boundary?
    # neg_indices = se3.nn(pos_indices).flatten().unique()
    # neg_indices = neg_indices[~torch.isin(neg_indices, pos_indices)]

    # Morphological gradient on positive indices
    # neighbours = se3.nn(pos_indices)
    # pos_indices = pos_indices[~torch.isin(neighbours, pos_indices.unsqueeze(0)).all(dim=1)]

    # cell_indices = torch.cat([pos_indices, neg_indices], dim=0)
    # labels = torch.cat([torch.ones_like(pos_indices), torch.zeros_like(neg_indices)], dim=0).bool()

    return cell_indices, labels


if __name__ == "__main__":
    from data_sampling.sample_morph import sample_morph
    from datetime import datetime

    torch.manual_seed(1)
    morphs = [sample_morph(1, i, True)[0] for i in range(1, 7)]
    start = datetime.now()
    labels, cell_indices = estimate_workspace(morphs[-1])
    print(f"Time: {datetime.now() - start}")
