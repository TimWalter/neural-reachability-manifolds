import warnings

warnings.filterwarnings("ignore", message=".*Dynamo detected a call to a `functools.lru_cache`.*")
import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Bool, Int64
from tabulate import tabulate

from data_sampling.autotune_batch_size import get_batch_size
from data_sampling.robotics import forward_kinematics, collision_check, analytical_inverse_kinematics, LINK_RADIUS, \
    transformation_matrix, get_capsules, EPS

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
    joint_limits = torch.zeros(morph.shape[0], 2, device=morph.device)

    extended_morph = torch.cat([torch.zeros_like(morph[:1]), morph])
    alpha0, a0, d0 = extended_morph[:-2].split(1, dim=-1)
    alpha1, a1, d1 = extended_morph[1:-1].split(1, dim=-1)

    coordinate_fix = torch.eye(4, device=morph.device, dtype=morph.dtype).repeat(morph.shape[0] - 1, 1, 1)
    wrist = (a1[:, 0] == 0) & (d1[:, 0] == 0)
    coordinate_fix[wrist] = transformation_matrix(alpha0, a0, d0, torch.zeros_like(d0))[wrist]

    plane_normal = torch.stack([
        torch.zeros_like(alpha1),
        -torch.sin(alpha1),
        torch.cos(alpha1),
        torch.zeros_like(alpha1)], dim=2)
    plane_anchor = torch.stack([
        a1,
        -d1 * torch.sin(alpha1),
        d1 * torch.cos(alpha1),
        torch.ones_like(alpha1)], dim=2)

    plane_normal = torch.sum(coordinate_fix * plane_normal, dim=-1)[:, :3]
    plane_anchor = torch.sum(coordinate_fix * plane_anchor, dim=-1)[:, :3]

    stacked_morph = torch.stack([extended_morph[:-2], extended_morph[1:-1], extended_morph[2:]], dim=1)
    stacked_morph[~wrist, 0, :] = 0.0
    start, end = get_capsules(stacked_morph, joints=torch.zeros(*stacked_morph.shape[:-1], 1, device=morph.device))
    capsules = end - start

    # Get closest non-zero capsule before joint
    pre_capsule = capsules[:, 3, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 2, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 1, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 0, :]

    # Get closest non-zero capsule after joint
    post_capsule = capsules[:, -2, :]
    post_capsule[mask] = capsules[mask := post_capsule.norm(dim=-1) < 1e-6, -1, :]

    in_plane = ((pre_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6
    in_plane &= ((post_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6

    limited = (pre_capsule.norm(dim=-1) > EPS) & (post_capsule.norm(dim=-1) > EPS) & in_plane

    mask = post_capsule.norm(dim=-1) > pre_capsule.norm(dim=-1)
    arc = torch.arcsin(2 * LINK_RADIUS / post_capsule.norm(dim=-1))
    arc[mask] = torch.arctan(2 * LINK_RADIUS / pre_capsule.norm(dim=-1))[mask]

    joint_limits[:-1, 0] = torch.where(limited, 2 * torch.pi - 2 * arc, 2 * torch.pi)  # Range
    angle = torch.atan2(torch.sum(torch.cross(pre_capsule, post_capsule, dim=1) * plane_normal, dim=1),
                        torch.sum(pre_capsule * post_capsule, dim=1))
    # if their angle becomes pi, they collide and are antiparallel
    angle = torch.atan2(torch.sin(torch.pi - angle), torch.cos(torch.pi - angle))
    joint_limits[:-1, 1] = torch.where(limited, angle + arc, -torch.pi)  # Offset

    return joint_limits


# @jaxtyped(typechecker=beartype)
def sample_reachable_poses(morph: Float[Tensor, "dof 3"], joint_limits: Float[Tensor, "batch_size dof 2"]) -> tuple[
    Float[Tensor, "n_valid 4 4"],
    Int64[Tensor, "n_valid"],
]:
    """
    Sample end effector poses for the robot and compute their discretised cell index.

    Args:
        morph: MDH parameters encoding the robot geometry.
        joint_limits: Joint limits that prevent most self-collisions.

    Returns:
        Reachable end effector poses and their respective cell indices.
    """
    joints = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + joint_limits[
        ..., 1:2]
    poses = forward_kinematics(morph, joints)
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
    centre = transformation_matrix(morph[0, 0:1], morph[0, 1:2], morph[0, 2:3], torch.zeros_like(morph[0, 2:3]))[:3, 3]
    radius = torch.sqrt(morph[:, 1] ** 2 + morph[:, 2] ** 2).sum() - morph[0, 2]

    return centre.cpu(), radius.item()


def estimate_capability_map(morph: Float[Tensor, "dofp1 3"], debug: bool = False) -> \
        Int64[Tensor, " num_samples"] | tuple[Int64[Tensor, " num_samples"], tuple[int, int, float, float, float]]:
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
    args = [morph.unsqueeze(0).expand(probe_size, -1, -1),
            joint_limits.unsqueeze(0).expand(probe_size, -1, -1)]
    batch_size = get_batch_size(morph.device, sample_reachable_poses, probe_size, args, safety=0.5)
    morph = morph.unsqueeze(0).expand(batch_size, -1, -1)
    joint_limits = joint_limits.unsqueeze(0).expand(batch_size, -1, -1)
    if debug:  # Warm-Up for benchmarking
        _ = compiled_sample_reachable_poses(morph, joint_limits)

    indices = []
    cuda_indices = []
    n_batches = 0
    start = datetime.now()
    while datetime.now() - start < timedelta(minutes=1):
        _, new_indices = compiled_sample_reachable_poses(morph, joint_limits)
        cuda_indices += [new_indices]
        n_batches += 1
        if len(cuda_indices) >= 1000:
            transfer = torch.cat(cuda_indices)
            pinned = torch.empty(transfer.shape, dtype=transfer.dtype, pin_memory=True)
            pinned.copy_(transfer, non_blocking=True)
            indices += [pinned]
            cuda_indices = []

    if len(cuda_indices) > 0:
        indices += [torch.cat(cuda_indices).cpu()]

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
    Int64[Tensor, " num_samples"],
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

    r_indices = estimate_capability_map(morph)

    centre, radius = estimate_reachable_ball(morph)
    cell_indices = se3.index(se3.random_ball(num_samples, centre, radius))
    labels = torch.isin(cell_indices, r_indices)

    return cell_indices, labels


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
