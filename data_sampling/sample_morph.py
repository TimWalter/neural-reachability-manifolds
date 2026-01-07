import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, Int, Bool, jaxtyped

from data_sampling.robotics import forward_kinematics, geometric_jacobian, yoshikawa_manipulability, collision_check, LINK_RADIUS


#@jaxtyped(typechecker=beartype)
def _sample_link_type(batch_size: int, dof: int) -> Int[Tensor, "batch_size {dof+1}"]:
    """
    Sample link types:
        0 <=> a≠0, d≠0 (General)
        1 <=> a=0, d≠0 (Only link length)
        2 <=> a≠0, d=0 (Only link offset)
        3 <=> a=0, d=0 (Point intersection / Spherical Wrist component)
    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots

    Returns:
        Link type sampled uniformly
    """
    link_type = torch.randint(0, 4, size=(batch_size, dof + 1))
    return link_type


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

    mask = (link_type == 1) | (link_type == 3)
    link_twist[mask] = link_twist_options[1 + (link_twist_choice[mask] > 1 / 2).to(torch.int64)]

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
    link_lengths[..., 0][(link_type == 1) | (link_type == 3)] = 0
    link_lengths[..., 1][(link_type == 2) | (link_type == 3)] = 0
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
    rejected = ((link_length.abs() < 2*LINK_RADIUS) & (link_length != 0)).any(dim=(1, 2))
    return rejected


#@jaxtyped(typechecker=beartype)
def _sample_morph(batch_size: int, dof: int, analytically_solvable: bool) -> Float[Tensor, "batch_size {dof+1} 3"]:
    """
    Sample morphologies, encoded as MDH parameters (alpha, a, d), given their respective rejection criteria.

    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots
        analytically_solvable: whether to sample only analytically solvable robots

    Returns:
        MDH parameters (alpha, a, d) describing the robot morphology
    """
    # 1. Sample link types
    link_types = _sample_link_type(batch_size, dof)

    # 2. Sample alphas
    link_twists = _sample_link_twist(link_types)
    while (mask := _reject_link_twist(link_twists)).any():
        link_twists[mask] = _sample_link_twist(link_types[mask])

    # 3. Sample link lengths
    link_lengths = _sample_link_length(link_types)
    while (mask := _reject_link_length(link_lengths)).any():
        link_lengths[mask] = _sample_link_length(link_types[mask])

    morph = torch.cat([link_twists.unsqueeze(-1), link_lengths], dim=2)

    if analytically_solvable:
        if dof == 5:  # Special 5 DOF (Mixed Parallel/Intersecting Axes)
            axes_choice = torch.randint(0, 2, (batch_size,))
            row_indices = torch.arange(batch_size).unsqueeze(-1)
            morph[row_indices, axes_choice + 2, 1] = 0
            morph[row_indices, 3 - axes_choice, 0] = 0
        elif dof == 6:  # Special 6 DOF (3 Parallel Inner Axes)
            axes_choice = torch.randint(2, 4, (batch_size,))
            indices_to_slice = axes_choice.unsqueeze(-1) + torch.arange(2)
            row_indices = torch.arange(batch_size).unsqueeze(-1)
            morph[row_indices, indices_to_slice, 0] = 0
        elif dof > 6:
            raise NotImplementedError("Analytically solvable sampling not implemented for DOF > 6")

    return morph


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
    rejected |= (yoshikawa_manipulability(jacobian, True) < 1e-4).all(dim=1)

    return rejected.cpu()


#@jaxtyped(typechecker=beartype)
def sample_morph(num_robots: int, dof: int, analytically_solvable: bool) -> Float[Tensor, "num_robots {dof+1} 3"]:
    """
   Sample valid morphologies, encoded in modified Denavit-Hartenberg parameters (alpha, a, d).

   Args:
       num_robots: number of robots to sample
       dof: degrees of freedom of the robots
       analytically_solvable: whether to sample only analytically solvable robots

   Returns:
       MDH parameters (alpha, a, d) describing the robot morphology
   """
    morph = _sample_morph(num_robots, dof, analytically_solvable)
    while (mask := _reject_morph(morph)).any():
        morph[mask] = _sample_morph(mask.sum().item(), dof, analytically_solvable)

    return morph

if __name__ == "__main__":
    sample_morph(num_robots=1, dof=6, analytically_solvable=False)
