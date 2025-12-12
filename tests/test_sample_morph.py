"""
Test specific combination of link type is valid.
"""

import torch
from data_sampling.robotics import forward_kinematics, geometric_jacobian
from data_sampling.sample_morph import sample_morph


def test_three_type_1_w_twist():
    num_rob = 50
    num_joint = 1000
    robots = torch.rand(num_rob, 7, 3)
    # twists are ±90 degrees except for the base
    robots[:, 0, 0] = 0
    robots[:, 1:, 0] = torch.pi/2
    robots[:, 1:, 0] *= (torch.randint(0, 2, robots[:, 1:, 0].shape) * 2 - 1)
    # All links are I-shape with a=0, d≠0
    robots[:, :, 1] = 0
    # Normalize the total length
    robots[:, :, 2] = robots[:, :, 2] / robots[:, :, 2].sum(dim=1, keepdim=True)

    joints = 2 * torch.pi * torch.rand(num_rob, num_joint, robots.shape[1], 1) - torch.pi
    joints[..., -1, :] = 0
    morph = robots.unsqueeze(1).expand(-1, num_joint, -1, -1)
    poses = forward_kinematics(morph, joints)
    jacobian = geometric_jacobian(poses)
    _, singular_values, _ = torch.linalg.svd(jacobian)
    manipulability = torch.prod(singular_values, dim=-1)
    assert (torch.all(manipulability.max(dim=0)[0]>2.5e-3) and torch.all(manipulability.min(dim=0)[0]<2.5e-4))
    assert ((manipulability>1e-3).sum() > num_rob*num_joint*0.4)


def test_two_type_1_no_twist():
    num_rob = 50
    num_joint = 1000
    # generate part of the kinematic chain
    latter_links = sample_morph(num_rob, 4)
    # first two links are type 1 without twist
    first_two_links = torch.zeros(num_rob, 2, 3)
    first_two_links[:, :, -1] = torch.rand(num_rob, 2)
    # twist the third joint to avoid three consecutive straight joints
    latter_links[:, 0, 0].masked_fill_(latter_links[:, 0, 0] == 0, torch.pi/2)
    latter_links[:, 0, 1].masked_fill_(latter_links[:, 0, 1] == 0, 0.12)
    robots = torch.cat((first_two_links, latter_links), dim=1)
    print(robots[:2])
    joints = 2 * torch.pi * torch.rand(num_rob, num_joint, robots.shape[1], 1) - torch.pi
    joints[..., -1, :] = 0
    morph = robots.unsqueeze(1).expand(-1, num_joint, -1, -1)
    poses = forward_kinematics(morph, joints)
    jacobian = geometric_jacobian(poses)
    _, singular_values, _ = torch.linalg.svd(jacobian)
    manipulability = torch.prod(singular_values, dim=-1)
    assert(torch.all(manipulability<1e-4))


def test_propostional_position():
    robot = sample_morph(1, 6).squeeze()
    # Corresponding links are proportional, here the ratio is 2
    similar_robot = torch.cat((robot[:,0:1], 2*robot[:, 1:]), dim=-1)
    joints = 2 * torch.pi * torch.rand(100, robot.shape[1], 1) - torch.pi
    joints[:, -1, :] = 0
    morph = robot.unsqueeze(0).expand(100, -1, -1)
    similar_morph = similar_robot.unsqueeze(0).expand(100, -1, -1)
    eef_poses = forward_kinematics(morph, joints)[:, -1, ...]
    similar_eef_poses = forward_kinematics(similar_morph, joints)[:, -1, ...]
    # orientation
    assert(torch.all(eef_poses[:, :3, :3]==similar_eef_poses[:, :3, :3]))
    # translational position
    assert(torch.all(2*eef_poses[:, :3, 3]==similar_eef_poses[:, :3, 3]))