import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, Int, Bool, jaxtyped

from data_sampling.robotics import forward_kinematics, geometric_jacobian, yoshikawa_manipulability, collision_check


#@jaxtyped(typechecker=beartype)
def _sample_link_type(batch_size: int, dof: int) -> Int[Tensor, "batch_size {dof+1}"]:
    """
    Sample link types
        0 <=> a≠0, d≠0
        1 <=> a=0, d≠0
        2 <=> a≠0, d=0
    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots

    Returns:
        Link type sampled uniformly
    """
    link_type = torch.randint(0, 3, size=(batch_size, dof + 1))
    return link_type


#@jaxtyped(typechecker=beartype)
def _reject_link_type(link_type: Int[Tensor, "batch_size dofp1"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link type that have three consecutive intersecting axes
    (i.e., a_i=0, a_{i+1}=0, a_{i+2}=0)

    Args:
        link_type: Link types to check

    Returns:
        Mask of rejected link types
    """
    rejected = ((link_type[:, :-2] == 1) & (link_type[:, 1:-1] == 1) & (link_type[:, 2:] == 1)).any(dim=1)
    return rejected


#@jaxtyped(typechecker=beartype)
def _sample_link_twist(link_type: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1"]:
    """
    Sample link twist angles (alphas) uniformly from {-pi/2, 0, pi/2}
    but avoid degenerate cases by preventing two consecutive collinear axes (alpha_i=0 & a_i=0)

    Args:
        link_type: Link types of the robots

    Returns:
        Link twist angles
    """
    link_twist_options = torch.tensor([0, torch.pi / 2, -torch.pi / 2])
    link_twist_choice = torch.rand(*link_type.shape)
    link_twist = link_twist_options[(link_twist_choice > 1 / 3).to(torch.int64) + (link_twist_choice > 2 / 3).to(torch.int64)]

    link_twist[link_type == 1] = link_twist_options[1 + (link_twist_choice[link_type == 1] > 1 / 2).to(torch.int64)]

    return link_twist


#@jaxtyped(typechecker=beartype)
def _reject_link_twist(link_twist: Float[Tensor, "batch_size dofp1"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject alphas that have more than two consecutive parallel axes
    (i.e., alpha_i=0, alpha_{i+1}=0, alpha_{i+2}=0)

    Args:
        link_twist: Link twist angles (alpha) to check

    Returns:
        Mask of rejected link twists
    """
    rejected = ((link_twist[:, :-2] == 0) & (link_twist[:, 1:-1] == 0) & (link_twist[:, 2:] == 0)).any(dim=1)
    return rejected


#@jaxtyped(typechecker=beartype)
def _sample_link_length(link_type: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1 2"]:
    """
    Sample link lengths uniformly on the simplex by normalizing exponential samples and splitting the lengths
    according to the link types and a random angle for the links with both a and d non-zero.

    Args:
        link_type: Link types of the robots

    Returns:
        Link lengths (a, d)
    """

    dist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
    link_lengths = dist.sample(torch.Size((*link_type.shape, 1)))[..., 0].repeat(1, 1, 2)
    link_lengths /= link_lengths.sum(dim=1, keepdim=True)
    gamma = torch.rand((link_type == 0).sum()) * 2 * torch.pi
    link_lengths[..., 0][link_type == 0] *= torch.sin(gamma)
    link_lengths[..., 1][link_type == 0] *= torch.cos(gamma)
    link_lengths[..., 0][link_type == 1] = 0
    link_lengths[..., 1][link_type == 2] = 0
    return link_lengths


#@jaxtyped(typechecker=beartype)
def _reject_link_length(link_length: Float[Tensor, "batch_size dofp1 2"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link lengths that are smaller than twice the link radius (0.025) but not zero.

    Args:
        link_length: Link lengths to check

    Returns:
        Mask of rejected link lengths
    """
    rejected = ((link_length.abs() < 0.05) & (link_length != 0)).any(dim=(1, 2))
    return rejected


#@jaxtyped(typechecker=beartype)
def _sample_morph(batch_size: int, dof: int) -> Float[Tensor, "batch_size {dof+1} 3"]:
    """
    Sample morphologies, encoded as MDH parameters (alpha, a, d), given their respective rejection criteria.

    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots

    Returns:
        MDH parameters (alpha, a, d) describing the robot morphology
    """
    # 1. Sample link types
    link_types = _sample_link_type(batch_size, dof)
    while (mask := _reject_link_type(link_types)).any():
        link_types[mask] = _sample_link_type(mask.sum().item(), dof)

    # 2. Sample alphas
    link_twists = _sample_link_twist(link_types)
    while (mask := _reject_link_twist(link_twists)).any():
        link_twists[mask] = _sample_link_twist(link_types[mask])

    # 3. Sample link lengths
    link_lengths = _sample_link_length(link_types)
    while (mask := _reject_link_length(link_lengths)).any():
        link_lengths[mask] = _sample_link_length(link_types[mask])

    return torch.cat([link_twists.unsqueeze(-1), link_lengths], dim=2)


#@jaxtyped(typechecker=beartype)
def _reject_morph(morph: Float[Tensor, "batch_size dofp1 3"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject morphologies that have adjacent links that seem constantly in collision or that seem
    degenerate (Jacobian is rank-deficient for all configurations).

    Args:
        morph: Morphologies to check

    Returns:
        Mask of rejected morphologies
    """
    joints = 2 * torch.pi * torch.rand(morph.shape[0], 1000, morph.shape[1], 1, device="cuda") - torch.pi
    morph = morph.unsqueeze(1).expand(-1, 1000, -1, -1).to("cuda")

    poses = forward_kinematics(morph, joints)
    rejected = collision_check(morph, poses, radius=0.025).all(dim=1)
    jacobian = geometric_jacobian(poses)
    rejected |= (yoshikawa_manipulability(jacobian) < 1e-4).all(dim=1)

    return rejected.cpu()


#@jaxtyped(typechecker=beartype)
def sample_morph(num_robots: int, dof: int) -> Float[Tensor, "num_robots {dof+1} 3"]:
    """
   Sample valid morphologies, encoded in modified Denavit-Hartenberg parameters (alpha, a, d).

   Args:
       num_robots: number of robots to sample
       dof: degrees of freedom of the robots

   Returns:
       MDH parameters (alpha, a, d) describing the robot morphology
   """
    morph = _sample_morph(num_robots, dof)
    while (mask := _reject_morph(morph)).any():
        morph[mask] = _sample_morph(mask.sum().item(), dof)

    return morph


if __name__ == "__main__":
    robots = sample_morph(10, 5)
    print(robots)
    print(robots.shape)
