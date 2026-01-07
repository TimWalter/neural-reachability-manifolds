import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int, Bool
from tabulate import tabulate

from data_sampling.autotune_batch_size import get_batch_size
from data_sampling.robotics import forward_kinematics, collision_check, analytical_inverse_kinematics, LINK_RADIUS

import data_sampling.se3 as se3
from datetime import datetime, timedelta

torch.set_float32_matmul_precision("high")


# @jaxtyped(typechecker=beartype)
def get_joint_limits(morph: Float[Tensor, "dof 3"]) -> Float[Tensor, "dof 2"]:
    """
    Compute joint limits based on the morphology to avoid self-collisions.

    Args:
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Joint limits
    """
    dof = morph.shape[0]
    joint_limits = torch.zeros(dof,2, device="cuda")

    alpha1, a1, d1 = morph[:-1].split(1, dim=-1)
    alpha2, a2, d2 = morph[1:].split(1, dim=-1)

    no_offset = d1 == 0
    overlapping_offset = ((a2.abs() - a1.abs() < 2 * LINK_RADIUS) &
                          (torch.sign(d1) != torch.sign(d2)) &
                          (d1.abs() - d2.abs() < 2 * LINK_RADIUS) &
                          (alpha2 == 0))
    limited = (a1 != 0) & (a2 != 0) & (no_offset | overlapping_offset)
    arc = torch.arcsin(2 * LINK_RADIUS / a2.abs())

    joint_limits[:-1, 0:1] = torch.where(limited, 2 * torch.pi - 2 * arc, 2 * torch.pi)
    joint_limits[:-1, 1:2] = torch.where(limited,
                                       torch.where(torch.sign(a1) == torch.sign(a2), -torch.pi + arc, arc),
                                       -torch.pi)

    return joint_limits

# @jaxtyped(typechecker=beartype)
def sample_joints(batch_size: int,
                  morph: Float[Tensor, "dof 3"],
                  joint_limits: Float[Tensor, "batch_size dof 2"]
                  ) -> Float[Tensor, "{batch_size} {dof}"]:
    """
    Sample random joint configurations for the robot.

    Args:
        batch_size: Number of joint configurations to sample
        morph: MDH parameters encoding the robot geometry.
        joint_limits: Joint limits to avoid self-collisions

    Returns:
        Newly sampled joint configurations
    """
    dof = morph.shape[0]
    joints = torch.rand(batch_size, dof, device="cuda") * joint_limits[..., 0] + joint_limits[..., 1]

    return joints


# @jaxtyped(typechecker=beartype)
def sample_reachable_poses(batch_size: int,
                           morph: Float[Tensor, "batch_size dof 3"],
                           joint_limits: Float[Tensor, "batch_size dof 2"]) -> tuple[
    Float[Tensor, "batch_size 4 4"],
    Int[Tensor, "batch_size"],
]:
    """
    Sample end effector poses for the robot and compute their discretised cell index.

    Args:
        batch_size: Number of end effector poses to sample.
        morph: MDH parameters encoding the robot geometry.
        joint_limits: Joint limits to avoid self-collisions

    Returns:
        Reachable end effector poses and their respective cell indices.
    """
    joints = sample_joints(batch_size, morph[0], joint_limits)
    poses = forward_kinematics(morph, joints.unsqueeze(-1))
    self_collision = collision_check(morph, poses)
    poses = poses[:, -1, :, :][~self_collision]
    cell_indices = se3.index(poses)
    return poses, cell_indices

compiled_sample_reachable_poses = torch.compile(sample_reachable_poses)


# @jaxtyped(typechecker=beartype)
def estimate_reachable_ball(morph: Float[Tensor, "dof 3"]) -> tuple[Float[Tensor, "3"], float]:
    """
    Estimate the reachable ball of a robot.

    Args:
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Center and radius of the reachable ball.
    """
    joint_limits = get_joint_limits(morph)
    probe_size = 2048
    args = [probe_size, morph.unsqueeze(0).expand(probe_size, -1, -1), joint_limits.unsqueeze(0).expand(probe_size, -1, -1)]
    batch_size = get_batch_size(morph.device, sample_reachable_poses, probe_size, args, safety=0.5)
    poses, _ = compiled_sample_reachable_poses(batch_size, morph.unsqueeze(0).expand(batch_size, -1, -1), joint_limits.unsqueeze(0).expand(batch_size, -1, -1))
    centre = torch.mean(poses[:, :3, 3], dim=0)
    radius = torch.norm(poses[:, :3, 3] - centre, dim=1).max().item()

    return centre, radius


@jaxtyped(typechecker=beartype)
def sample_capability_map_analytically(morph: Float[Tensor, "dof 3"], num_samples: int) -> tuple[
    Float[Tensor, " {num_samples} 4 4"],
    Bool[Tensor, " {num_samples}"]
]:
    """
    Estimate the workspace of a robot with analytically solvable inverse kinematics.

    Args:
        morph: MDH parameters encoding the robot geometry
        num_samples: Number of samples to generate
    Returns:
        Poses and labels encoding the discretised capability map
    """
    morph = morph.to("cuda")

    centre, radius = estimate_reachable_ball(morph)
    poses = se3.random_ball(num_samples, centre, radius)

    joints, manipulability = analytical_inverse_kinematics(morph, poses)
    labels = manipulability.cpu() != -1

    return poses.cpu(), labels


def estimate_capability_map(morph: Float[Tensor, "dofp1 3"], debug: bool = False) -> \
        Int[Tensor, " num_samples"] | tuple[Int[Tensor, " num_samples"], tuple[int, int, float, float, float]]:
    """
    Estimat the capability map using only forward kinematics, a discretisation of SE(3) and the closed world assumption.
    Fill up the discretised cells using FK until convergence. All unfilled cells are assumed to be unreachable.

    Args:
        morph: MDH parameters encoding the robot geometry.
        debug: Whether to return benchmark parameters.

    Returns:
        Cell indices of reachable cells and optionally benchmark parameters.
    """
    joint_limits = get_joint_limits(morph)
    probe_size = 2048
    args = [probe_size, morph.unsqueeze(0).expand(probe_size, -1, -1), joint_limits.unsqueeze(0).expand(probe_size, -1, -1)]
    batch_size = get_batch_size(morph.device, sample_reachable_poses, probe_size, args, safety=0.5)
    morph = morph.unsqueeze(0).expand(batch_size, -1, -1)
    joint_limits = joint_limits.unsqueeze(0).expand(batch_size, -1, -1)
    if debug:  # Warm-Up for benchmarking
        _ = compiled_sample_reachable_poses(batch_size, morph, joint_limits)

    indices = []
    n_batches = 0
    start = datetime.now()
    while datetime.now() - start < timedelta(minutes=1):
        _, new_indices = compiled_sample_reachable_poses(batch_size, morph, joint_limits)
        indices += [new_indices.cpu()]
        n_batches += 1

    indices = torch.cat(indices)
    collision_free_samples = indices.shape[0]
    indices = indices.unique()

    if debug:
        filled_cells = indices.shape[0]
        total_samples = n_batches * batch_size
        total_efficiency = filled_cells / total_samples
        unique_efficiency = filled_cells / collision_free_samples
        collision_efficiency = collision_free_samples / total_samples
        return indices, (filled_cells, total_samples, total_efficiency, unique_efficiency, collision_efficiency)
    return indices


# @jaxtyped(typechecker=beartype)
def sample_capability_map(morph: Float[Tensor, "dofp1 3"], num_samples: int) -> tuple[
    Int[Tensor, " num_samples"],
    Bool[Tensor, " num_samples"]
]:
    """
    Estimate the workspace of a robot solely from forward kinematics.

    Args:
        morph: MDH parameters encoding the robot geometry.
        num_samples: Number of samples to generate.

    Returns:
        Labels and indices encoding the discretised capability map
    """
    morph = morph.to("cuda")

    centre, radius = estimate_reachable_ball(morph)
    cell_indices = se3.index(se3.random_ball(num_samples, centre, radius))

    r_indices = estimate_capability_map(morph)
    labels = torch.isin(cell_indices, r_indices)

    return cell_indices, labels


if __name__ == "__main__":
    from data_sampling.sample_morph import sample_morph

    torch.manual_seed(1)
    morphs = sample_morph(10, 6, True)
    benchmarks = []
    for morph in morphs:
        morph = morph.to("cuda")
        _, benchmark = estimate_capability_map(morph, True)
        benchmarks += [torch.tensor(benchmark)]

    mean_benchmark = torch.stack(benchmarks).mean(dim=0, keepdim=True).tolist()
    mean_benchmark[0][0] = int(mean_benchmark[0][0])
    mean_benchmark[0][1] = int(mean_benchmark[0][1])
    print(tabulate(mean_benchmark,
                   headers=["Filled Cells", "Total Samples\n(Speed)", "Efficiency\n(Total)", "Efficiency\n(Unique)",
                            "Efficiency\n(Collision)"], floatfmt=".4f", intfmt=","))
