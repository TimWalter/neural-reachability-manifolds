import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int, Bool

from data_sampling.autotune_batch_size import get_batch_size
from data_sampling.robotics import forward_kinematics, collision_check, analytical_inverse_kinematics, unique_with_index

import data_sampling.se3 as se3

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
def sample_reachable_poses(batch_size: int, morph: Float[Tensor, "dof 3"]) -> tuple[
    Float[Tensor, "n_valid 4 4"],
    Int[Tensor, "n_valid"]
]:
    """
    Sample end effector poses for the robot and compute their discretised cell index.

    Args:
        batch_size: Number of end effector poses to sample.
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Reachable end effector poses and their respective cell indices.
    """
    joints = sample_joints(batch_size, morph.shape[0])
    poses = forward_kinematics(morph.unsqueeze(0).expand(batch_size, -1, -1), joints.unsqueeze(-1))
    self_collision = collision_check(morph.unsqueeze(0).expand(poses.shape[0], -1, -1), poses)
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
    args = (morph,)
    batch_size = get_batch_size(morph.device, sample_reachable_poses, args, safety=0.5)
    poses, _ = compiled_sample_reachable_poses(batch_size, *args)
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


def estimate_capability_map(morph: Float[Tensor, "dofp1 3"]) -> Int[Tensor, " num_samples"]:
    """
    Estimat the capability map using only forward kinematics, a discretisation of SE(3) and the closed world assumption.
    Fill up the discretised cells using FK until convergence. All unfilled cells are assumed to be unreachable.

    Args:
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Cell indices of reachable cells.
    """
    args = (morph,)
    batch_size = get_batch_size(morph.device, sample_reachable_poses, args, safety=0.5)

    r_indices = torch.empty(0, dtype=torch.int64, device="cuda")
    prev_shape = r_indices.shape[0]
    while r_indices.shape[0] == 0 or (1 - prev_shape / r_indices.shape[0]) > 1e-5:
        prev_shape = r_indices.shape[0]

        _, new_r_indices = compiled_sample_reachable_poses(batch_size, *args)
        r_indices = torch.cat([r_indices, new_r_indices]).unique()

    r_indices = r_indices.cpu()

    # Dilate
    #r_indices = torch.cat([r_indices, se3.nn(r_indices).flatten()], dim=0).unique()
    print(r_indices.shape[0]//1e6)
    return r_indices


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
    from datetime import datetime

    torch.manual_seed(1)
    morphs = [sample_morph(1, i, True)[0] for i in range(6, 7)]
    start = datetime.now()
    labels, cell_indices = sample_capability_map(morphs[-1], 100_000)
    print(f"Time: {datetime.now() - start}")
