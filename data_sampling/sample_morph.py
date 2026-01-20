import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, Int, Bool, jaxtyped

from data_sampling.robotics import forward_kinematics, geometric_jacobian, yoshikawa_manipulability, collision_check, \
    LINK_RADIUS


# @jaxtyped(typechecker=beartype)
def _sample_link_type(batch_size: int, dof: int) -> Int[Tensor, "batch_size {dof+1}"]:
    """
    Sample link types:
        0 <=> a≠0, d≠0 (General)
        1 <=> a=0, d≠0 (Only link length)
        2 <=> a≠0, d=0 (Only link offset)
        3 <=> a=0, d=0 (Spherical wrist component) - Half as likely since it also affects the former joint (for joint limits)
    Args:
        batch_size: number of robots to sample
        dof: degrees of freedom of the robots

    Returns:
        Link type sampled uniformly
    """
    link_type = torch.randint(0, 7, size=(batch_size, dof + 1)) // 2
    link_type[:, :-1][link_type[:, 1:] == 3] = 1 + 2 * torch.randint(0, 1, size=link_type[:, :-1][link_type[:, 1:] == 3].shape)

    return link_type


def _reject_link_type(link_type: Int[Tensor, "batch_size dofp1"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link type combinations that are not manufacturable.

    Args:
        link_type: Link types of the robots

    Returns:
        Mask of rejected link types
    """
    rejected = ((link_type[:, 2:] == 3) & (link_type[:, 1:-1] == 3) & (link_type[:, :-2] == 3)).any(dim=1)
    return rejected


# @jaxtyped(typechecker=beartype)
def _sample_link_twist(link_type: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1"]:
    """
    Sample link twist angles (α) uniformly from {-π/2, 0, π/2}

    Args:
        link_type: Link types of the robots

    Returns:
        Link twist angles
    """
    link_twist_options = torch.tensor([0, torch.pi / 2, -torch.pi / 2])
    link_twist_choice = torch.randint(0, 3, size=link_type.shape)
    link_twist = link_twist_options[link_twist_choice]
    return link_twist


# @jaxtyped(typechecker=beartype)
def _reject_link_twist(link_twist: Float[Tensor, "batch_size dofp1"], link_type: Int[Tensor, "batch_size dofp1"]) \
        -> Bool[Tensor, "batch_size"]:
    """
    Reject link twists that lead to degeneracy by having
    more than two consecutive parallel axes (i.e. α_i=0, α_{i+1}=0, α_{i+2}=0),
    collinear axes (i.e. a Type2 joint by another Type2 Joint with α_{i+1}=0),
    or that do describe unmanufacturable spherical wrist joints.

    Args:
        link_twist: Link twist angles (alpha) to check
        link_type: Link types of the robots

    Returns:
        Mask of rejected link twists
    """
    rejected = ((link_twist[:, :-2] == 0) & (link_twist[:, 1:-1] == 0) & (link_twist[:, 2:] == 0)).any(dim=1)
    rejected |= ((link_twist[:, :-2] == 0) & (link_type[:, 1:-1] == 3) & (link_twist[:, 2:] == 0)).any(dim=1)

    rejected |= ((link_type[:, :-1] == 2) & (link_type[:, 1:] == 2) & (link_twist[:, 1:] == 0)).any(dim=1)

    rejected |= ((link_type == 3) & (link_twist == 0)).any(dim=1)
    return rejected


def _sample_analytically_solvable_link_types_and_twist(batch_size: int, dof: int) \
        -> tuple[Int[Tensor, "batch_size dofp1"], Float[Tensor, "batch_size dofp1"]]:
    link_type = _sample_link_type(batch_size, dof)
    link_twist = _sample_link_twist(link_type)
    if dof == 5:
        types = torch.randint(0, 3, size=(batch_size,))
        """
        5 DOF Analytically solvable robot types:
            0 <=> (a_0=0) | (a_4=0)                 - (The last or first two axes intersect)
            1 <=> (a_2=0 & α_4=0) | (a_4=0 & α_2=0) - (One pair of consecutive, intermediate axes intersects while the other is parallel)
            2 <=> α_i=0 & α_{i+1}=0                 - (Any three consecutive axes are parallel)
        """
        link_type[types == 0, torch.randint(0, 2, ((types == 0).sum().item(),)) * 4] = 1

        axes_choice = torch.randint(0, 2, ((types == 1).sum().item(),))
        link_type[types == 1, axes_choice * 2 + 2] = 1
        link_twist[types == 1, (~axes_choice.bool()) * 2 + 2] = 0

        axes_choice = torch.randint(0, 4, ((types == 2).sum().item(),))
        link_twist[types == 2, axes_choice] = 0
        link_twist[types == 2, axes_choice + 1] = 0

    elif dof == 6:
        types = torch.randint(0, 1, size=(batch_size,))
        """
        6 DOF Analytically solvable robot types:
            0 <=> (a_4=0 & a_5=0 & d_4=0 & d_5=0) | (a_1=0 & a_2=0 & d_1=0 & d_1=0) - (Spherical wrist at the beginning or end)
            1 <=> (α_1=0 & α_2=0 & a_5=0) | (α_4=0 & α_5=0 & a_1=0)                 - (3 Parallel & 2 intersecting axes on opposing ends)
            2 <=> (α_2=0 & α_3=0) | (α_3=0 & α_4=0)                                 - (3 Parallel inner axes)
        """
        axes_choice = torch.randint(0, 2, ((types == 0).sum().item(),))
        link_type[types == 0, 1 + 3 * axes_choice] = 3
        link_type[types == 0, 2 + 3 * axes_choice] = 3

        axes_choice = torch.randint(0, 2, ((types == 1).sum().item(),))
        link_type[types == 1, 1 + 4 * axes_choice] = 1
        link_twist[types == 1, 1 + 3 * (~axes_choice.bool())] = 0
        link_twist[types == 1, 2 + 3 * (~axes_choice.bool())] = 0

        axes_choice = torch.randint(2, 4, ((types == 2).sum().item(),))
        link_twist[types == 2, axes_choice] = 0
        link_twist[types == 2, axes_choice + 1] = 0

    elif dof > 6:
        raise NotImplementedError("Analytically solvable sampling not implemented for DOF > 6")
    return link_type, link_twist


# @jaxtyped(typechecker=beartype)
def _sample_link_length(link_type: Int[Tensor, "batch_size dofp1"]) -> Float[Tensor, "batch_size dofp1 2"]:
    """
    Sample link lengths uniformly on the simplex by normalising exponential samples and splitting the lengths
    according to the link types and a random angle for the links with both a and d non-zero.

    Args:
        link_type: Link types of the robots

    Returns:
        Link lengths (a, d)
    """

    dist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
    link_lengths = dist.sample(torch.Size((*link_type.shape, 1)))[..., 0].repeat(1, 1, 2)
    link_lengths /= (link_lengths * (link_type.unsqueeze(-1) != 3)).sum(dim=1, keepdim=True)

    gamma = torch.rand((link_type == 0).sum()) * 2 * torch.pi
    link_lengths[..., 0][link_type == 0] *= torch.sin(gamma)
    link_lengths[..., 1][link_type == 0] *= torch.cos(gamma)

    link_lengths[..., 0][(link_type == 1) | (link_type == 3)] = 0
    link_lengths[..., 1][(link_type == 2) | (link_type == 3)] = 0
    return link_lengths


# @jaxtyped(typechecker=beartype)
def _reject_link_length(link_length: Float[Tensor, "batch_size dofp1 2"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject link lengths that are smaller than twice the link radius (0.025) but not zero.

    Args:
        link_length: Link lengths to check

    Returns:
        Mask of rejected link lengths
    """
    rejected = ((link_length.abs() < 2 * LINK_RADIUS) & (link_length != 0)).any(dim=(1, 2))
    return rejected


# @jaxtyped(typechecker=beartype)
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
    if analytically_solvable:
        link_types, link_twists = _sample_analytically_solvable_link_types_and_twist(batch_size, dof)
        while (mask := _reject_link_type(link_types) | _reject_link_twist(link_twists, link_types)).any():
            link_types[mask], link_twists[mask] = _sample_analytically_solvable_link_types_and_twist(mask.sum().item(),
                                                                                                     dof)
    else:
        link_types = _sample_link_type(batch_size, dof)
        while (mask := _reject_link_type(link_types)).any():
            link_types[mask] = _sample_link_type(mask.sum().item(), dof)

        link_twists = _sample_link_twist(link_types)
        while (mask := _reject_link_twist(link_twists, link_types)).any():
            link_twists[mask] = _sample_link_twist(link_types[mask])

    link_lengths = _sample_link_length(link_types)
    while (mask := _reject_link_length(link_lengths)).any():
        link_lengths[mask] = _sample_link_length(link_types[mask])

    morph = torch.cat([link_twists.unsqueeze(-1), link_lengths], dim=2)

    return morph


# @jaxtyped(typechecker=beartype)
def _reject_morph(morph: Float[Tensor, "batch_size dofp1 3"]) -> Bool[Tensor, "batch_size"]:
    """
    Reject morphologies that have adjacent links that seem constantly in collision or that seem
    degenerate (Jacobian is rank-deficient for all configurations).

    Args:
        morph: Morphologies to check

    Returns:
        Mask of rejected morphologies
    """
    joints = 2 * torch.pi * torch.rand(morph.shape[0], 1000, morph.shape[1], 1) - torch.pi
    morph = morph.unsqueeze(1).expand(-1, 1000, -1, -1)

    poses = forward_kinematics(morph, joints)
    rejected = collision_check(morph, poses).all(dim=1)
    jacobian = geometric_jacobian(poses)
    rejected |= (yoshikawa_manipulability(jacobian, True) < 1e-4).all(dim=1)

    return rejected


# @jaxtyped(typechecker=beartype)
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
