import torch
from data_sampling.robotics import forward_kinematics, collision_check, LINK_RADIUS

def test_immediate_self_collision():
    a_list = torch.rand(100) * 0.9 - 0.45
    a_list += torch.sign(a_list) * 2 * LINK_RADIUS
    for a in a_list:
        temp = torch.tensor([
            [0.0, 0.5, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, 0.05]
        ])
        arc = torch.arcsin(2*LINK_RADIUS/a.abs())
        range = 2*torch.pi - 2*arc
        if a > 0: # same sign
            offset = -torch.pi + arc
        else: # different sign
            offset = arc

        joints = torch.linspace(-torch.pi, torch.pi, 1000)
        if offset + range > torch.pi:
            prediction = (joints < offset) & (joints > torch.atan2(torch.sin(offset + range), torch.cos(offset + range)))
        else:
            prediction = (joints < offset) | (joints > offset + range)
        joints = torch.stack([joints, torch.zeros_like(joints), torch.zeros_like(joints)], dim=1)

        pose = forward_kinematics(temp.unsqueeze(0).expand(1000, -1, -1), joints.unsqueeze(-1))

        collision = collision_check(temp.unsqueeze(0).expand(1000, -1, -1), pose)

        assert (collision == prediction).all()
