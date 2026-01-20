import torch
from data_sampling.sample_morph import sample_morph
from data_sampling.sample_capability_map import get_joint_limits
from data_sampling.robotics import forward_kinematics, collision_check


def test_scissor_collision():
    normal_morphs = sample_morph(100, 6, False)
    analytical_morphs = sample_morph(100, 6, True)
    for morphs in [normal_morphs, analytical_morphs]:
        print("Morphs Type")
        for morph_idx, morph in enumerate(morphs):
            joint_limits = get_joint_limits(morph)

            extended_morph = torch.cat([torch.zeros_like(morph[:1]), morph])
            alpha0, a0, d0 = extended_morph[:-2].split(1, dim=-1)
            alpha1, a1, d1 = extended_morph[1:-1].split(1, dim=-1)
            alpha2, a2, d2 = extended_morph[2:].split(1, dim=-1)
            wrist = (a1[:, 0] == 0) & (d1[:, 0] == 0)
            limited = joint_limits[:-1, 0] != 2 * torch.pi
            for joint_idx in range(morph.shape[0] - 1):

                if limited[joint_idx]:
                    isolated_morph = morph[joint_idx - (1 if wrist[joint_idx] else 0):joint_idx + 2, :].clone()
                    if a2[joint_idx] != 0:
                        isolated_morph[-1, 2] = 0
                    if wrist[joint_idx]:
                        if d0[joint_idx] != 0:
                            isolated_morph[0, 1] = 0
                    else:
                        if d1[joint_idx] != 0:
                            isolated_morph[0, 1] = 0
                    collision_detector = torch.tensor([[0.0, 0.0, 1e-2]])
                    isolated_morph = torch.cat([collision_detector, isolated_morph, collision_detector], dim=0)

                    isolated_morph = isolated_morph.unsqueeze(0).expand(100, -1, -1)
                    # Account for the collision detector
                    damped_joint_limits = joint_limits[joint_idx].clone()
                    damped_joint_limits[0] -= torch.pi / 10
                    damped_joint_limits[1] += torch.pi / 20
                    non_colliding_joints = torch.rand(100, isolated_morph.shape[1], 1,
                                                      device=morph.device) * damped_joint_limits[0:1] + damped_joint_limits[
                                               1:2]

                    poses = forward_kinematics(isolated_morph, non_colliding_joints)
                    critical_distance = collision_check(isolated_morph, poses, debug=True)
                    assert (critical_distance > -5e-3).all(), \
                        f"{morph_idx}, {joint_idx} {(critical_distance < 0.0).nonzero()}"

                    colliding_joints = torch.zeros(2, isolated_morph.shape[1], 1)
                    colliding_joints[0, :] = 1.0
                    colliding_joints = colliding_joints * joint_limits[joint_idx, 0:1] + joint_limits[joint_idx, 1:2]
                    over_edge = torch.zeros_like(colliding_joints)
                    over_edge[0, :] = torch.pi / 20
                    over_edge[1, :] = - torch.pi / 20
                    colliding_joints += over_edge
                    isolated_morph = isolated_morph[0].unsqueeze(0).expand(2, -1, -1)
                    poses = forward_kinematics(isolated_morph, colliding_joints)
                    critical_distance = collision_check(isolated_morph, poses, debug=True)
                    assert (critical_distance < 0.0).all(), f"{morph_idx}, {joint_idx} {critical_distance}"


