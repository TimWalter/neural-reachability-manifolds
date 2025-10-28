import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool, jaxtyped
from beartype import beartype

from data_sampling.robotics import forward_kinematics, geometric_jacobian, yoshikawa_manipulability, collision_check


def generate_eaik_conform_mdhs(num_robots: int, robot_class: int) -> Float[Tensor, "num_robots dof 3"]:
    """
    Generate manipulators that are analytically solveable according to
    D. Ostermeier, J. Külz and M. Althoff, "Automatic Geometric Decomposition for Analytical Inverse Kinematics,"
    in IEEE Robotics and Automation Letters, vol. 10, no. 10, pp. 9964-9971, Oct. 2025, doi: 10.1109/LRA.2025.3597897.

    Args:
        num_robots: number of robots to generate
        robot_class: robot class according to the paper

    Returns:
        Robot geometries in MDH parameters (alpha, a, d)
    """
    if robot_class == 3:  # <5 DOF #TODO not working in EAIK yet (raise Issue)
        mdh = sample_mdh(num_robots, dof=4)

    elif robot_class == 5:  # Special 5 DOF
        mdh = sample_mdh(num_robots, dof=5)
        condition = torch.randint(0, 3, (num_robots,))

        for i in range(3):
            robot_indices = torch.where(condition == i)[0]
            axes_choice = torch.randint(0, 2 if i != 2 else 3, robot_indices.shape)
            for idx, choice in zip(robot_indices, axes_choice):
                if i == 0:  # first/last two axes intersect
                    mdh[idx, choice * 3 + 1, 1] = 0
                elif i == 1:  # Mixed Parallel/Intersecting Axes
                    mdh[idx, choice + 2, 1] = 0
                    mdh[idx, 3 - choice, 0] = 0
                else:  # Any three consecutive axes are parallel
                    mdh[idx, 1 + choice:3 + choice, 0] = 0

    elif robot_class == 6:  # Spherical Wrist 6 DOF # TODO Test
        mdh = sample_mdh(num_robots, dof=6)
        mdh[:, -2:, 0][torch.isclose(mdh[:, -2:, 0], torch.tensor([0.]), atol=.01)] = torch.pi / 2
        mdh[:, -2:, 1] = 0
        mdh[:, -1, 2] = 0
        # Avoid four intersecting axes
        # mdh[:, -3, 1][torch.isclose(mdh[:, -3, 1], torch.tensor([0.]), atol=.01)] = min_length

    elif robot_class == 7:  # 3 Parallel & 2 Intersecting Axes on opposing ends 6 DOF #TODO incorrect
        mdh = sample_mdh(num_robots, dof=6)
        axes_choice = torch.randint(0, 1, (num_robots,))
        for robot_idx, axes in enumerate(axes_choice):
            mdh[robot_idx, axes * 4:axes * 4 + 2, 0] = 0
            mdh[robot_idx, 5 - axes * 5, 1] = 0
            mdh[robot_idx, 6, 2] = 0

    elif robot_class == 8:  # 3 Parallel inner axes 6 DOF
        mdh = sample_mdh(num_robots, dof=6)
        axes_choice = torch.randint(2, 4, (num_robots,))
        for robot_idx, axes in enumerate(axes_choice):
            mdh[robot_idx, axes:axes + 2, 0] = 0

    else:
        raise ValueError(f"Unsupported robot class {robot_class}")

    # return normalise_mdh(mdh)


@jaxtyped(typechecker=beartype)
def _sample_link_types(batch_size: int, dof: int) -> Int[Tensor, "batch_size {dof+1}"]:
    """
    Sample link types
        0 <=> a≠0, d≠0
        1 <=> a=0, d≠0
        2 <=> a≠0, d=0
    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots

    Returns:
        Link types sampled uniformly
    """
    link_types = torch.randint(0, 3, size=(batch_size, dof + 1))
    return link_types


@jaxtyped(typechecker=beartype)
def _reject_link_types(link_types: Int[Tensor, "batch_size dofp1"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link types that have three consecutive intersecting axes
    (i.e., a_i=0, a_{i+1}=0, a_{i+2}=0)

    Args:
        link_types: Link types to check

    Returns:
        Mask of rejected link types
    """
    rejected = ((link_types[:, :-2] == 1) & (link_types[:, 1:-1] == 1) & (link_types[:, 2:] == 1)).any(dim=1)
    return rejected


@jaxtyped(typechecker=beartype)
def _sample_alphas(link_types: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1"]:
    """
    Sample link twist angles (alphas) uniformly from {-pi/2, 0, pi/2}
    but avoid degenerate cases by preventing two consecutive collinear axes (alpha_i=0 & a_i=0)

    Args:
        link_types: Link types of the robots

    Returns:
        Link twist angles
    """
    alpha_options = torch.tensor([0, torch.pi / 2, -torch.pi / 2])
    alpha_choice = torch.rand(*link_types.shape)
    alphas = alpha_options[(alpha_choice > 1 / 3).to(torch.int64) + (alpha_choice > 2 / 3).to(torch.int64)]

    alphas[link_types == 1] = alpha_options[1 + (alpha_choice[link_types == 1] > 1 / 2).to(torch.int64)]

    return alphas


@jaxtyped(typechecker=beartype)
def _reject_alphas(alphas: Float[Tensor, "batch_size dofp1"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject alphas that have more than two consecutive parallel axes
    (i.e., alpha_i=0, alpha_{i+1}=0, alpha_{i+2}=0)

    Args:
        alphas: Link twist angles to check

    Returns:
        Mask of rejected alphas
    """
    rejected = ((alphas[:, :-2] == 0) & (alphas[:, 1:-1] == 0) & (alphas[:, 2:] == 0)).any(dim=1)
    return rejected


@jaxtyped(typechecker=beartype)
def _sample_link_lengths(link_types: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1 2"]:
    """
    Sample link lengths uniformly on the simplex by normalizing exponential samples and splitting the lengths
    according to the link types and a random angle for the links with both a and d non-zero.

    Args:
        link_types: Link types of the robots

    Returns:
        Link lengths (a, d)
    """

    dist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
    link_lengths = dist.sample(torch.Size((*link_types.shape, 1)))[..., 0].repeat(1, 1, 2)
    link_lengths /= link_lengths.sum(dim=1, keepdim=True)
    gamma = torch.rand((link_types == 0).sum()) * 2 * torch.pi
    link_lengths[..., 0][link_types == 0] *= torch.sin(gamma)
    link_lengths[..., 1][link_types == 0] *= torch.cos(gamma)
    link_lengths[..., 0][link_types == 1] = 0
    link_lengths[..., 1][link_types == 2] = 0
    return link_lengths


@jaxtyped(typechecker=beartype)
def _reject_link_lengths(link_lengths: Float[Tensor, "batch_size dofp1 2"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link lengths that are smaller than twice the link radius (0.025) but not zero.

    Args:
        link_lengths: Link lengths to check

    Returns:
        Mask of rejected link lengths
    """
    rejected = ((link_lengths.abs() < 0.05) & (link_lengths != 0)).any(dim=(1, 2))
    return rejected


@jaxtyped(typechecker=beartype)
def _sample_mdh(batch_size: int, dof: int) -> Float[Tensor, "batch_size {dof+1} 3"]:
    """
    Sample MDH parameters (alpha, a, d) given their respective rejection criteria.

    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots

    Returns:
        MDH parameters (alpha, a, d) describing the robot geometries
    """
    # 1. Sample link types
    link_types = _sample_link_types(batch_size, dof)
    while (mask := _reject_link_types(link_types)).any():
        link_types[mask] = _sample_link_types(mask.sum().item(), dof)

    # 2. Sample alphas
    alphas = _sample_alphas(link_types)
    while (mask := _reject_alphas(alphas)).any():
        alphas[mask] = _sample_alphas(link_types[mask])

    # 3. Sample link lengths
    link_lengths = _sample_link_lengths(link_types)
    while (mask := _reject_link_lengths(link_lengths)).any():
        link_lengths[mask] = _sample_link_lengths(link_types[mask])

    return torch.cat([alphas.unsqueeze(-1), link_lengths], dim=2)


@jaxtyped(typechecker=beartype)
def _reject_mdh(mdhs: Float[Tensor, "batch_size dofp1 3"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject MDH parameters that have adjacent links that seem constantly in collision or that seem
    degenerate (Jacobian is rank-deficient for all configurations).

    Args:
        mdh: MDH parameters to check

    Returns:
        Mask of rejected MDH parameters
    """
    joints = 2 * torch.pi * torch.rand(mdhs.shape[0], 1000, mdhs.shape[1], 1, device="cuda") - torch.pi
    mdh = mdhs.unsqueeze(1).expand(-1, 1000, -1, -1).to("cuda")

    poses = forward_kinematics(mdh, joints)
    rejected = collision_check(mdh, poses, radius=0.025).all(dim=1)
    jacobian = geometric_jacobian(poses)
    rejected |= (yoshikawa_manipulability(jacobian) < 1e-4).all(dim=1)

    return rejected.cpu()


@jaxtyped(typechecker=beartype)
def sample_mdh(num_robots: int, dof: int) -> Float[Tensor, "num_robots {dof+1} 3"]:
    """
   Sample valid MDH parameters (alpha, a, d).

   Args:
       num_robots: number of robots to sample
       dof: degrees of freedom of the robots

   Returns:
       MDH parameters (alpha, a, d) describing the robot geometries
   """
    mdh = _sample_mdh(num_robots, dof)
    while (mask := _reject_mdh(mdh)).any():
        mdh[mask] = _sample_mdh(mask.sum().item(), dof)

    return mdh


if __name__ == "__main__":
    robots = sample_mdh(10, 5)
    print(robots)
    print(robots.shape)
