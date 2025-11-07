import torch
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped
from torch import Tensor

LINK_RADIUS = 0.025
@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
def yoshikawa_manipulability(jacobian: Float[Tensor, "*batch 6 dof"]) -> Float[Tensor, "*batch"]:
    """
    Computes the Yoshikawa manipulability index using the SVD of the Jacobian.

    Args:
        jacobian: Geometric Jacobians for each robot configuration.

    Returns:
        Manipulability index.
    """
    _, singular_values, _ = torch.linalg.svd(jacobian)
    manipulability = torch.prod(singular_values[..., : min(6, jacobian.shape[-1])], dim=-1)

    return manipulability

@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
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
    idx = torch.arange(2 * dof, device=device)
    pairs = torch.combinations(idx, r=2)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    # Skip capsule pairs from the same joint or adjacent joint
    same_joint_mask = (i_idx % dof == j_idx % dof)
    adjacent_joint_mask = (torch.abs((i_idx // dof) - (j_idx // dof)) == 1) & \
                          ((i_idx % dof) + (j_idx % dof) == dof)
    mask = same_joint_mask | adjacent_joint_mask
    pairs = pairs[~mask]
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
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
