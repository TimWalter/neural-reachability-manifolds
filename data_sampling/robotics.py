from typing import Optional

import torch
import numpy as np
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped, Int
from torch import Tensor
from eaik.IK_Homogeneous import HomogeneousRobot
from data_sampling.se3 import distance
from scipy.spatial.transform import Rotation

LINK_RADIUS = 0.025
EPS = 1e-4


# @jaxtyped(typechecker=beartype)
def transformation_matrix(alpha: Float[Tensor, "*batch 1"], a: Float[Tensor, "*batch 1"], d: Float[Tensor, "*batch 1"],
                          theta: Float[Tensor, "*batch 1"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Computes the modified Denavit-Hartenberg transformation matrix.

    Args:
        alpha: Twist angle
        a: Link length
        d: Link offset
        theta: Joint angle

    Returns:
        Transformation matrix.
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    ct, st = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)
    return torch.stack([torch.cat([ct, -st, zero, a], dim=-1),
                        torch.cat([st * ca, ct * ca, -sa, -d * sa], dim=-1),
                        torch.cat([st * sa, ct * sa, ca, d * ca], dim=-1),
                        torch.cat([zero, zero, zero, one], dim=-1)], dim=-2)


# @jaxtyped(typechecker=beartype)
def forward_kinematics(mdh: Float[Tensor, "*batch dofp1 3"],
                       theta: Float[Tensor, "*batch dofp1 1"]) -> Float[Tensor, "*batch dofp1 4 4"]:
    """
    Computes forward kinematics for a robot defined by modified Denavit-Hartenberg parameters.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        theta: The joint angle (theta_i) for each joint.

    Returns:
        The transformation matrices from the base frame to each joint frame.
    """
    transforms = transformation_matrix(mdh[..., 0:1], mdh[..., 1:2], mdh[..., 2:3], theta)

    poses = []
    pose = torch.eye(4, device=mdh.device, dtype=mdh.dtype).expand(*mdh.shape[:-2], 1, 4, 4)
    for i in range(mdh.shape[-2]):
        poses.append(pose := pose @ transforms[..., i:i + 1, :, :])

    return torch.cat(poses, dim=-3)


# @jaxtyped(typechecker=beartype)
def geometric_jacobian(poses: Float[Tensor, "*batch dofp1 4 4"]) -> Float[Tensor, "*batch 6 dof"]:
    """
    Compute the geometric Jacobian for revolute joints in the base frame.

    Args:
        poses: Homogeneous transforms (batch, dof, 4, 4) from base to each joint, including the end-effector
               as the last transform.

    Returns:
        Jacobian where first 3 rows are linear velocity components and last 3 rows are angular velocity components.
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
        soft: If true assume workspace dimension is equal to dof.

    Returns:
        Manipulability index.
    """
    if soft and jacobian.shape[-1] < 6:
        _, singular_values, _ = torch.linalg.svd(jacobian)
        manipulability = torch.prod(singular_values[..., : min(6, jacobian.shape[-1])], dim=-1)
    else:
        manipulability = torch.sqrt(torch.det(torch.matmul(jacobian, jacobian.transpose(-1, -2))).abs())
    return manipulability


def unique_with_index(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim,
                                           sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, index


# @jaxtyped(typechecker=beartype)
def unique_indices(indices: Int[Tensor, "batch"],
                   manipulability: Float[Tensor, "batch"],
                   other: list[Float[Tensor, "batch *rest"]]) \
        -> tuple[
            Int[Tensor, "n_unique"],
            Float[Tensor, "n_unique"],
            list[Float[Tensor, "n_unique *rest"]]
        ]:
    """
    Select unique indices, keeping the ones with the highest manipulability.

    Args:
        indices: Indices corresponding to determine uniqueness
        manipulability: Manipulability values corresponding to indices
        other: Other tensors to be filtered accordingly

    Returns:
        Unique indices, their manipulability, and other tensors filtered accordingly.
    """
    manipulability, sort_indices = torch.sort(manipulability, descending=True)
    indices = indices[sort_indices]
    other = [tensor[sort_indices] for tensor in other]

    indices, unique_indices = unique_with_index(indices)

    manipulability = manipulability[unique_indices]
    other = [tensor[unique_indices] for tensor in other]

    return indices, manipulability, other


# @jaxtyped(typechecker=beartype)
def signed_distance_capsule_capsule(s1: Float[Tensor, "*batch 3"], e1: Float[Tensor, "*batch 3"], r1: float,
                                    s2: Float[Tensor, "*batch 3"], e2: Float[Tensor, "*batch 3"], r2: float) \
        -> Float[Tensor, "*batch"]:
    l1 = e1 - s1
    l2 = e2 - s2
    ds = s1 - s2

    alpha = (l1 * l1).sum(dim=-1, keepdim=True)
    beta = (l2 * l2).sum(dim=-1, keepdim=True)
    gamma = (l1 * l2).sum(dim=-1, keepdim=True)
    delta = (l1 * ds).sum(dim=-1, keepdim=True)
    epsilon = (l2 * ds).sum(dim=-1, keepdim=True)

    det = alpha * beta - gamma ** 2

    t1 = torch.clamp((gamma * epsilon - beta * delta) / (det + 1e-10), 0.0, 1.0)
    t2 = torch.clamp((gamma * t1 + epsilon) / (beta + 1e-10), 0.0, 1.0)

    t1 = torch.where((t2 == 0.0) | (t2 == 1.0), torch.clamp((t2 * gamma - delta) / (alpha + 1e-10), 0.0, 1.0), t1)

    c1 = s1 + t1 * l1
    c2 = s2 + t2 * l2

    point_distance = ((c1 - c2) ** 2).sum(dim=-1)

    return point_distance - (r1 + r2) ** 2


# @jaxtyped(typechecker=beartype)
def get_capsules(mdh: Float[torch.Tensor, "*batch dofp1 3"],
                 poses: Optional[Float[torch.Tensor, "*batch dofp1 4 4"]] = None,
                 joints: Optional[Float[torch.Tensor, "*batch dofp1 1"]] = None) -> tuple[
    Float[torch.Tensor, "*batch 2*dofp1 3"],
    Float[torch.Tensor, "*batch 2*dofp1 3"],
]:
    *batch_shape, dof, _ = mdh.shape
    device = mdh.device
    dtype = mdh.dtype

    # Prepend the Identity matrix (Base Frame 0) to poses (poses currently contains [T1, T2, ..., TN]. We need [T0, T1, ..., TN].)
    identity = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 1, 4, 4)
    if poses is None:
        if joints is None:
            joints = torch.zeros((*batch_shape, dof, 1), device=device, dtype=dtype)
        poses = forward_kinematics(mdh, joints)
    poses = torch.cat([identity, poses], dim=-3)

    # Starting point of the first capsule (a) is the pose from base
    s_a = poses[..., :-1, :3, 3]
    # End point of the second capsule (d) is the pose after
    e_d = poses[..., 1:, :3, 3]
    # The middle point is deducted by reversing the translation d along the z-axis
    z_axis = poses[..., 1:, :3, 2]
    d = mdh[..., 2].unsqueeze(-1)
    e_a = s_d = e_d - d * z_axis

    # Assemble the chain (stack+flatten essentially zips such that s_a_1, s_d_1, s_a_2, s_d_2, ..)
    s_all = torch.stack([s_a, s_d], dim=-2).flatten(-3, -2)
    e_all = torch.stack([e_a, e_d], dim=-2).flatten(-3, -2)
    return s_all, e_all

PAIR_COMBINATIONS = [torch.triu_indices(2 * dof, 2 * dof, offset=2) for dof in range(1, 8)]


# #@jaxtyped(typechecker=beartype)
def collision_check(mdh: Float[torch.Tensor, "*batch dofp1 3"],
                    poses: Float[torch.Tensor, "*batch dofp1 4 4"],
                    radius: float = LINK_RADIUS,
                    debug=False) -> Bool[torch.Tensor, "*batch"] | Float[torch.Tensor, "*batch"]:
    """
    Compute whether the robot is in self-collision for each batch element.

    Args:
        mdh: Modified DH parameters [alpha, a, d, theta]
        poses: Homogeneous transforms for each joint (world frame)
        radius: Capsule radius
        debug: Whether to return the relevant signed distances directly
    Returns:
        A boolean indicating whether each configuration is in collision.
    """
    global PAIR_COMBINATIONS
    if mdh.device != PAIR_COMBINATIONS[0].device:
        PAIR_COMBINATIONS = [pair.to(mdh.device) for pair in PAIR_COMBINATIONS]

    *batch_shape, dof, _ = mdh.shape

    s_all, e_all = get_capsules(mdh, poses)

    # Capsule pair combinations
    i_idx, j_idx = PAIR_COMBINATIONS[dof - 1]
    num_pairs = i_idx.shape[0]

    # Gather capsule endpoints
    s1 = s_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    e1 = e_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    s2 = s_all[..., j_idx, :].reshape(-1, num_pairs, 3)
    e2 = e_all[..., j_idx, :].reshape(-1, num_pairs, 3)

    # Ignore distances with missing capsules
    collisions = (torch.norm(s1 - e1, dim=-1) > EPS) & (torch.norm(s2 - e2, dim=-1) > EPS)
    # Ignore adjacent capsules
    collisions &= torch.norm(e1 - s2, dim=-1) > EPS
    # Compute signed distances
    distances = signed_distance_capsule_capsule(s1, e1, radius, s2, e2, radius)
    if not debug:
        collisions &= distances < -EPS
        return collisions.any(dim=-1).reshape(batch_shape)
    else:
        return (distances*collisions).min(dim=-1).values.reshape(batch_shape)


# @jaxtyped(typechecker=beartype)
def inverse_kinematics(mdh: Float[Tensor, "dofp1 3"],
                       poses: Float[Tensor, "batch 4 4"]) -> tuple[
    Float[Tensor, "batch dofp1 1"],
    Float[Tensor, "batch"]
]:
    """
    Computes inverse kinematics for a robot defined by modified Denavit-Hartenberg parameters.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        poses: The desired poses (homogeneous transforms) to find joint angles for.

    Returns:
        The joint angles (theta_i) for each joint that achieve the desired poses with the highest manipulability and
        the manipulability values.
    """
    try:
        joints, manipulability = analytical_inverse_kinematics(mdh, poses)
    except RuntimeError as e:
        print(f"{e} \n Falling back numerical IK.")
        joints, manipulability = numerical_inverse_kinematics(mdh, poses)

    return joints, manipulability


def pure_analytical_inverse_kinematics(mdh: Float[Tensor, "dofp1 3"], poses: Float[Tensor, "batch 4 4"]) -> list[
    Float[Tensor, "n_solutions dofp1 1"],
]:
    """
    Computes inverse kinematics for a robot defined by modified Denavit-Hartenberg parameters analytically via EAIK and
    returns all solutions without checking for self-collisions or whether we actually end up in the correct position.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        poses: The desired poses (homogeneous transforms) to find joint angles for.

    Returns:
        The joint solutions
    """
    local_coord = transformation_matrix(mdh[:, 0:1], mdh[:, 1:2], mdh[:, 2:3], torch.zeros_like(mdh[:, 2:3]))
    global_coords = torch.empty_like(local_coord)
    global_coords[0] = local_coord[0]
    for i in range(1, len(local_coord)):
        global_coords[i] = global_coords[i - 1] @ local_coord[i]
    joint_transforms = torch.cat((global_coords, global_coords[-1].unsqueeze(0)), dim=0)
    eaik_bot = HomogeneousRobot(joint_transforms.cpu().numpy(), fixed_axes=[(mdh.shape[0] - 1, 0.0)])
    if not eaik_bot.hasKnownDecomposition():
        raise RuntimeError(f"Robot is not analytically solvable. {mdh}")
    solutions = eaik_bot.IK_batched(poses.cpu().numpy())
    joints = [torch.from_numpy(sol.Q.copy()).to(mdh.device).unsqueeze(-1) if sol.num_solutions() != 0
              else torch.empty(0, mdh.shape[0], 1, device=mdh.device, dtype=mdh.dtype)
              for sol in solutions]

    return joints


# @jaxtyped(typechecker=beartype)
def analytical_inverse_kinematics(mdh: Float[Tensor, "dofp1 3"], poses: Float[Tensor, "batch 4 4"]) -> tuple[
    Float[Tensor, "batch dofp1 1"],
    Float[Tensor, "batch"]
]:
    """
    Computes inverse kinematics for a robot defined by modified Denavit-Hartenberg parameters analytically via EAIK.

    Args:
        mdh: Contains [alpha_i, a_i, d_i] for each joint.
        poses: The desired poses (homogeneous transforms) to find joint angles for.

    Returns:
        The joint angles (theta_i) for each joint that achieve the desired poses with the highest manipulability and
        the manipulability values.
    """
    joints = pure_analytical_inverse_kinematics(mdh, poses)
    pose_indices = torch.cat(
        [torch.full((joint.shape[0], 1), i, dtype=torch.int64) for i, joint in enumerate(joints)], dim=0).to(mdh.device)
    joints = torch.cat(joints, dim=0).to(mdh.device)
    if joints.shape[0] != 0:
        bmorph = mdh.unsqueeze(0).expand(joints.shape[0], -1, -1).to(mdh.device)
        full_poses = forward_kinematics(bmorph, joints)
        self_collision = collision_check(bmorph, full_poses)

        pose_error = distance(full_poses[:, -1, :, :], poses[pose_indices[:, 0]]).squeeze(-1)

        mask = ~self_collision & (pose_error < EPS)
        full_poses = full_poses[mask]
        joints = joints[mask]
        pose_indices = pose_indices[mask]

        jacobian = geometric_jacobian(full_poses)
        manipulability = yoshikawa_manipulability(jacobian)

        pose_indices, manipulability, [joints] = unique_indices(pose_indices[:, 0], manipulability, [joints])
    else:
        pose_indices = torch.zeros((*poses.shape[:-2],), device=mdh.device, dtype=torch.bool)
        manipulability = torch.empty(0, device=mdh.device, dtype=mdh.dtype)

    full_joints = torch.zeros((*poses.shape[:-2], mdh.shape[0], 1), device=mdh.device, dtype=mdh.dtype)
    full_joints[pose_indices] = joints

    full_manipulability = -torch.ones((*poses.shape[:-2],), device=mdh.device, dtype=mdh.dtype)
    full_manipulability[pose_indices] = manipulability

    return full_joints, full_manipulability


# @jaxtyped(typechecker=beartype)
def numerical_inverse_kinematics(mdh: Float[Tensor, "dofp1 3"], poses: Float[Tensor, "batch 4 4"]) -> tuple[
    Float[Tensor, "batch dofp1 1"],
    Float[Tensor, "batch"]
]:
    """
    Numerical IK using damped least-squares.

    Args:
        mdh: Modified DH parameters [alpha_i, a_i, d_i].
        poses: Target poses [batch, 4, 4].

    Returns:
        Joint angles and manipulability values.
    """
    bmorph = mdh.unsqueeze(0).expand(poses.shape[0], -1, -1)

    joints_current = 2 * torch.pi * torch.rand(poses.shape[0], mdh.shape[0] - 1, 1, device=mdh.device) - torch.pi
    joints_best = joints_current.clone()
    min_errors = torch.ones(poses.shape[0], device=mdh.device) * torch.inf

    for i in range(200):
        full_joints = torch.cat([joints_current, torch.zeros(poses.shape[0], 1, 1, device=mdh.device)], dim=1)
        reached_pose = forward_kinematics(bmorph, full_joints)
        pose_error = distance(reached_pose[:, -1, :, :], poses).squeeze(-1)

        improvement = pose_error < min_errors
        min_errors[improvement] = pose_error[improvement]
        joints_best[improvement] = joints_current[improvement]

        jacobian = geometric_jacobian(reached_pose)

        rhs = torch.concat([
            poses[..., :3, 3] - reached_pose[:, -1, :3, 3],
            Rotation.from_matrix(poses[..., :3, :3] @ reached_pose[:, -1, :3, :3].transpose(-2, -1)).as_rotvec()
        ], dim=-1)

        try:
            update = torch.linalg.lstsq(jacobian, rhs, rcond=1e-3)[0].clip(-torch.pi, torch.pi)
        except torch.linalg.LinAlgError:
            w = torch.eye(6, device=mdh.device) * (1 + 1e-6)
            jacobian = (jacobian.transpose(-2, -1) @ w @ jacobian +
                        1e-3 * torch.eye(jacobian.shape[-1], device=mdh.device)).inverse() @ jacobian.transpose(-2,
                                                                                                                -1) @ w
            update = torch.einsum('bjx,bx->bj', jacobian, rhs)

        joints_current = torch.pi + joints_current + 0.2 * update.unsqueeze(2)
        joints_current = torch.atan2(torch.sin(joints_current), torch.cos(joints_current))

    joints = torch.cat([joints_best, torch.zeros(poses.shape[0], 1, 1, device=mdh.device)], dim=1)
    reached_pose = forward_kinematics(bmorph, joints)
    jacobian = geometric_jacobian(reached_pose)
    manipulability = yoshikawa_manipulability(jacobian)

    self_collision = collision_check(bmorph, reached_pose)
    pose_err = distance(reached_pose[:, -1, :, :], poses).squeeze(-1)
    full_mask = (pose_err < EPS) & ~self_collision

    manipulability[~full_mask] = -1

    return joints, manipulability
