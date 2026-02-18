import torch

from neural_capability_maps.dataset.morphology import sample_morph, get_joint_limits
from neural_capability_maps.dataset.self_collision import collision_check
from neural_capability_maps.dataset.kinematics import forward_kinematics

torch.set_default_dtype(torch.float64)

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
                if not limited[joint_idx]:
                    continue
                isolated_morph = morph[joint_idx - (1 if wrist[joint_idx] else 0):joint_idx + 2, :].clone()
                if a2[joint_idx] != 0:
                    isolated_morph[-1, 2] = 0
                if wrist[joint_idx]:
                    if d0[joint_idx] != 0:
                        isolated_morph[0, 1] = 0
                else:
                    if d1[joint_idx] != 0:
                        isolated_morph[0, 1] = 0

                isolated_morph = isolated_morph.unsqueeze(0).expand(100, -1, -1)
                non_colliding_joints = torch.rand(100, isolated_morph.shape[1], 1,
                                                  device=morph.device) * joint_limits[joint_idx][0:1] + joint_limits[joint_idx][1:2]

                poses = forward_kinematics(isolated_morph, non_colliding_joints)
                critical_distance = collision_check(isolated_morph, poses, debug=True)
                assert (critical_distance >= 0.0).all(), \
                    f"{isolated_morph[0]} \n {non_colliding_joints[torch.argmin(critical_distance)]}"

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
                assert (critical_distance < 0.0).all(), f"{isolated_morph[0]} \n {colliding_joints[torch.argmax(critical_distance)]}"


