import torch
import numpy as np
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped, Int
from torch import Tensor
from eaik.IK_Homogeneous import HomogeneousRobot
from data_sampling.se3 import distance
from scipy.spatial.transform import Rotation


LINK_RADIUS = 0.025
EPS = 1e-3


#@jaxtyped(typechecker=beartype)
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



#@jaxtyped(typechecker=beartype)
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
    pose = torch.eye(4, device=mdh.device).expand(*mdh.shape[:-2], 1, 4, 4)
    for i in range(mdh.shape[-2]):
        poses.append(pose := pose @ transforms[..., i:i + 1, :, :])

    return torch.cat(poses, dim=-3)


#@jaxtyped(typechecker=beartype)
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


#@jaxtyped(typechecker=beartype)
def yoshikawa_manipulability(jacobian: Float[Tensor, "*batch 6 dof"]) -> Float[Tensor, "*batch"]:
    """
    Computes the Yoshikawa manipulability index using the SVD of the Jacobian.

    Args:
        jacobian: Geometric Jacobians for each robot configuration.

    Returns:
        Manipulability index.
    """
    manipulability = torch.sqrt(torch.det(torch.matmul(jacobian, jacobian.transpose(-1, -2))).abs())
    # .abs() protects against tiny negative floating point errors
    return manipulability


# #@jaxtyped(typechecker=beartype)
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

    indices, inverse, counts = torch.unique(indices, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    unique_indices = inv_sorted[tot_counts]

    manipulability = manipulability[unique_indices]
    other = [tensor[unique_indices] for tensor in other]

    return indices, manipulability, other


# #@jaxtyped(typechecker=beartype)
def signed_distance_capsule_capsule(s1: Float[Tensor, "batch 3"], e1: Float[Tensor, "batch 3"], r1: float,
                                    s2: Float[Tensor, "batch 3"], e2: Float[Tensor, "batch 3"], r2: float) \
        -> Float[Tensor, "batch"]:
    l1 = e1 - s1
    l2 = e2 - s2
    ds = s1 - s2

    alpha = (l1 * l1).sum(dim=1, keepdim=True)
    beta = (l2 * l2).sum(dim=1, keepdim=True)
    gamma = (l1 * l2).sum(dim=1, keepdim=True)
    delta = (l1 * ds).sum(dim=1, keepdim=True)
    epsilon = (l2 * ds).sum(dim=1, keepdim=True)

    det = alpha * beta - gamma ** 2

    # Unconstrained solution
    t1 = (gamma * epsilon - beta * delta) / (det + 1e-10)
    t2 = (gamma * delta - alpha * epsilon) / (det + 1e-10)

    p1_interior = s1 + t1 * l1
    p2_interior = s2 + t2 * l2
    d_interior = (p1_interior - p2_interior).norm(dim=1, keepdim=True)

    # Endpoint–segment candidates
    t_s1 = torch.clamp(((s1 - s2) * l2).sum(dim=1, keepdim=True) / (beta + 1e-10), 0, 1)

    p_s1 = s2 + t_s1 * l2
    d_s1 = (s1 - p_s1).norm(dim=1, keepdim=True)

    t_e1 = torch.clamp(((e1 - s2) * l2).sum(dim=1, keepdim=True) / (beta + 1e-10), 0, 1)
    p_e1 = s2 + t_e1 * l2
    d_e1 = (e1 - p_e1).norm(dim=1, keepdim=True)

    t_s2 = torch.clamp(((s2 - s1) * l1).sum(dim=1, keepdim=True) / (alpha + 1e-10), 0, 1)
    p_s2 = s1 + t_s2 * l1
    d_s2 = (s2 - p_s2).norm(dim=1, keepdim=True)

    t_e2 = torch.clamp(((e2 - s1) * l1).sum(dim=1, keepdim=True) / (alpha + 1e-10), 0, 1)
    p_e2 = s1 + t_e2 * l1
    d_e2 = (e2 - p_e2).norm(dim=1, keepdim=True)

    # Combine endpoint candidates
    d_endpoints = torch.stack([d_s1, d_e1, d_s2, d_e2]).min(dim=0).values

    # Piecewise definition
    point_distance = torch.where((det == 0) | (t1 < 0) | (t1 > 1) | (t2 < 0) | (t2 > 1),
                                 d_endpoints,
                                 d_interior).squeeze(dim=1)

    return point_distance - r1 - r2


v_signed_distance = torch.vmap(signed_distance_capsule_capsule, in_dims=(0, 0, None, 0, 0, None))


# #@jaxtyped(typechecker=beartype)
def collision_check(mdh: Float[torch.Tensor, "*batch dofp1 3"],
                    poses: Float[torch.Tensor, "*batch dofp1 4 4"],
                    radius: float = LINK_RADIUS) -> Bool[torch.Tensor, "*batch"]:
    """
    Compute whether the robot is in self-collision for each batch element.

    Args:
        mdh: Modified DH parameters [alpha, a, d, theta]
        poses: Homogeneous transforms for each joint (world frame)
        radius: Capsule radius

    Returns:
        A boolean indicating whether each configuration is in collision.
    """
    *batch_shape, dof, _ = mdh.shape
    device = mdh.device

    # Extract joint positions and local axes in world frame
    origin = poses[..., :3, 3]  # (*batch, dof, 3)
    x_axis = poses[..., :3, 0]
    z_axis = poses[..., :3, 2]

    # DH parameters
    a = mdh[..., 1]
    d = mdh[..., 2]

    # Capsule endpoints
    s_a = origin
    e_a = origin + a.unsqueeze(-1) * x_axis
    s_d = origin
    e_d = origin + d.unsqueeze(-1) * z_axis

    # Stack both capsule families
    s_all = torch.cat([s_a, s_d], dim=-2)  # (*batch, 2*dof, 3)
    e_all = torch.cat([e_a, e_d], dim=-2)

    # Capsule pair combinations
    i_idx, j_idx = torch.triu_indices(2 * dof, 2 * dof, offset=1, device=device)

    # Skip capsule pairs from the same joint or adjacent joint
    same_joint_mask = (i_idx % dof == j_idx % dof)
    adjacent_joint_mask = (torch.abs((i_idx // dof) - (j_idx // dof)) == 1) & \
                          ((i_idx % dof) + (j_idx % dof) == dof)
    mask = same_joint_mask | adjacent_joint_mask
    i_idx = i_idx[~mask]
    j_idx = j_idx[~mask]
    num_pairs = len(i_idx)

    # Gather capsule endpoints
    s1 = s_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    e1 = e_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    s2 = s_all[..., j_idx, :].reshape(-1, num_pairs, 3)
    e2 = e_all[..., j_idx, :].reshape(-1, num_pairs, 3)

    # Compute signed distances
    distances = v_signed_distance(s1, e1, radius, s2, e2, radius)  # (batch_flat, num_pairs)

    # Collision if any signed distance < 0 in that configuration
    collision_flags = (distances < 0).any(dim=-1)

    return collision_flags.reshape(batch_shape)


#@jaxtyped(typechecker=beartype)
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
        print("Not analytically solvable, falling back numerical IK")
        joints, manipulability = numerical_inverse_kinematics(mdh, poses)

    return joints, manipulability


#@jaxtyped(typechecker=beartype)
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
    local_coord = transformation_matrix(mdh[:, 0:1], mdh[:, 1:2], mdh[:, 2:3], torch.zeros_like(mdh[:, 2:3]))
    global_coords = torch.empty_like(local_coord)
    global_coords[0] = local_coord[0]
    for i in range(1, len(local_coord)):
        global_coords[i] = global_coords[i - 1] @ local_coord[i]
    joint_transforms = torch.cat((global_coords, global_coords[-1].unsqueeze(0)), dim=0)
    eaik_bot = HomogeneousRobot(joint_transforms.cpu().numpy(), fixed_axes=[(mdh.shape[0] - 1, 0.0)])
    if not eaik_bot.hasKnownDecomposition():
        raise RuntimeError("Robot is not analytically solvable.")
    solutions = eaik_bot.IK_batched(poses.numpy())
    joints = [sol.Q for sol in solutions if sol.num_solutions() != 0]
    if not joints:
        raise RuntimeError("No analytical IK solutions found for any pose.")
    joints = np.concat(joints, axis=0)
    joints = torch.from_numpy(joints).float().unsqueeze(2).to(mdh.device)
    pose_indices = torch.cat(
        [torch.full((sol.num_solutions(), 1), i, dtype=torch.int64) for i, sol in enumerate(solutions)], dim=0)
    bmorph = mdh.unsqueeze(0).expand(joints.shape[0], -1, -1).to(mdh.device)
    desired_pose = poses[pose_indices[:, 0]].to(mdh.device)
    pose_indices = pose_indices.to(mdh.device)
    reached_pose = forward_kinematics(bmorph, joints)
    self_collision = collision_check(bmorph, reached_pose)
    pose_err = distance(reached_pose[:, -1, :, :], desired_pose).squeeze()
    full_mask = (pose_err < EPS) & ~self_collision

    reached_pose = reached_pose[full_mask]
    joints = joints[full_mask]
    pose_indices = pose_indices[full_mask]

    jacobian = geometric_jacobian(reached_pose)
    manipulability = yoshikawa_manipulability(jacobian)

    pose_indices, manipulability, [joints] = unique_indices(pose_indices[:, 0], manipulability, [joints])

    full_joints = torch.zeros((*poses.shape[:-2], mdh.shape[0], 1), device=mdh.device)
    full_joints[pose_indices] = joints

    full_manipulability = -torch.ones((*poses.shape[:-2],), device=mdh.device)
    full_manipulability[pose_indices] = manipulability

    return full_joints, full_manipulability


#@jaxtyped(typechecker=beartype)
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
