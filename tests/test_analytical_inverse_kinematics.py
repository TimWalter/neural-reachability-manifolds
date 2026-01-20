import torch

import neural_capability_maps.dataset.se3 as se3
from neural_capability_maps.dataset.self_collision import collision_check, EPS
from neural_capability_maps.dataset.kinematics import pure_analytical_inverse_kinematics, analytical_inverse_kinematics, forward_kinematics
from neural_capability_maps.dataset.morphology import sample_morph, get_joint_limits

torch.set_printoptions(sci_mode=False, precision=2)
torch.set_default_dtype(torch.float64)

def test_pure_analytical_inverse_kinematics():
    torch.manual_seed(1)
    n_samples = 1000
    n_robots = 10
    morphs = sample_morph(n_robots, 6, True)
    for morph_idx, morph in enumerate(morphs):
        joint_limits = get_joint_limits(morph).unsqueeze(0).expand(n_samples, -1, -1)
        morph = morph.unsqueeze(0).expand(n_samples, -1, -1)
        joints = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + joint_limits[
            ..., 1:2]
        poses = forward_kinematics(morph, joints)[:, -1, :, :]
        solutions = pure_analytical_inverse_kinematics(morph[0], poses)

        for i, solution in enumerate(solutions):
            if solution.shape[0] == 0:
                assert False, (f"IK failed to find a solution for pose \n{poses[i]}\n, "
                               f"joints \n{joints[i, :, 0]}\n and morph \n{morph[0]}\n")

            dist = torch.arctan2(torch.sin(solution[..., 0] - joints[i, :, 0]), torch.cos(solution[..., 0] - joints[i, :, 0]))
            dist = torch.norm(dist, dim=-1)
            assert torch.min(dist).item() < torch.pi / 100, (f"IK failed to reconstruct the joints \n {joints[i, :, 0]}\n"
                                                             f"It only found \n{solution[..., 0]}\n"
                                                             f"Instead of reaching pose \n{poses[i]}\n"
                                                             f"It reached only \n{forward_kinematics(morph[0].unsqueeze(0).expand(solution.shape[0], -1, -1), solution)[:, -1, :, :]}\n"
                                                             f"For morph \n{morph[0]}\n"
                                                             )


def test_analytical_inverse_kinematics():
    torch.manual_seed(1)
    n_samples = 1000
    n_robots = 10
    morphs = sample_morph(n_robots, 6, True)

    for i, morph in enumerate(morphs):
        joint_limits = get_joint_limits(morph).unsqueeze(0).expand(n_samples, -1, -1)
        morph = morph.unsqueeze(0).expand(n_samples, -1, -1)
        joints = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + joint_limits[
            ..., 1:2]
        poses = forward_kinematics(morph, joints)
        self_collision = collision_check(morph, poses)
        poses = poses[:, -1, :, :]
        joints, manipulability = analytical_inverse_kinematics(morph[0], poses)

        mask = manipulability != -1
        assert torch.all(self_collision[~mask]), (f"IK does not find solutions for joints "
                                                  f"\n{joints[~self_collision[~mask]]}\n "
                                                  f"and poses"
                                                  f"\n{poses[~self_collision[~mask]]}\n"
                                                  f"given morph"
                                                  f"\n{morph[0]}\n")

        morph = morph[0].unsqueeze(0).expand(mask.sum(), -1, -1)
        ik_poses = forward_kinematics(morph, joints[mask])
        assert se3.distance(poses[mask], ik_poses[:, -1, :, :]).max() < EPS, f"IK finds wrong solutions"
        ik_self_collisions = collision_check(morph, ik_poses)
        assert torch.all(~ik_self_collisions), f"IK solution has self-collisions"
