import argparse
import os
import re
from pathlib import Path

import torch
import pyarrow.parquet

from tqdm import tqdm

from robot import batch_kinematic_jacobian, compute_manipulability_svd, Robot
from transforms import ParameterConvention
from transforms.dh_conventions import mdh_to_homogeneous

from eaik.IK_Homogeneous import HomogeneousRobot
from estimate_workspace import estimate_workspace
from mdh_generation import generate_eaik_conform_mdhs

# Sample MDHs conformant with EAIK
# Sample Poses
# Get Joints via IK
# No-Solution -> Not reachable -> label = 0
# Joints -> Jacobian -> Manipulability Index = label

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available, CUDA used")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
torch.set_default_device(device)

save_folder = Path(__file__).parent.parent / 'data' / 'validated_test_set'
save_folder.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_robots", type=int, default=10, help="number of robots to generate")
args = parser.parse_args()
kwargs = vars(args)

real_robots = torch.tensor([
    [  # UR5
        [0, 0, 0.08946],
        [torch.pi / 2, 0, 0],
        [0, 0.425, 0],
        [0, 0.3922, 0.1091],
        [-torch.pi / 2, 0, 0.09465],
        [torch.pi / 2, 0, 0.0823],
    ],
])

mdhs = real_robots  # generate_eaik_conform_mdhs(kwargs['num_robots'] - 1, 3)

successful_robots = 0
pbar = tqdm(mdhs, desc=f"Generating validated data")
for mdh in pbar:
    local_trafos = mdh_to_homogeneous(torch.cat([mdh, torch.zeros(mdh.shape[0], 1)], dim=1))
    global_trafos = torch.empty_like(local_trafos)
    global_trafos[0] = local_trafos[0]
    for i in range(1, len(local_trafos)):
        global_trafos[i] = global_trafos[i - 1] @ local_trafos[i]
    joint_transforms = torch.cat((global_trafos, global_trafos[-1].unsqueeze(0)), dim=0)
    try:
        eaik_bot = HomogeneousRobot(joint_transforms.cpu().numpy())
    except RuntimeError as e:
        print(e)
        print(mdh)
        exit()

    poses, labels, joints = estimate_workspace(mdh, full_poses=True)

    if labels.sum() == -3240*3112:
        pbar.set_description(f"Skipping robot - no valid configurations found")
        continue

    seemingly_unreachable_poses = poses[labels == 0]
    if seemingly_unreachable_poses.shape[0] != 0:
        # print(mdh.shape[0])
        # print(joint_transforms)
        # print(mdh)
        # print(seemingly_unreachable_poses[:10])
        solutions = eaik_bot.IK_batched(seemingly_unreachable_poses.cpu().numpy())

        new_labels = torch.zeros(seemingly_unreachable_poses.shape[0], dtype=torch.float32)
        for i, sol in enumerate(solutions):
            joints = torch.from_numpy(sol.Q.copy()).to(torch.float32).to(torch.get_default_device())
            mask = torch.from_numpy(sol.is_LS.copy()).to(torch.bool).to(torch.get_default_device())
            if mask.any():
                dks_bot = Robot(mdh, ParameterConvention.MDH)
                jacobian = batch_kinematic_jacobian(joint_offsets=dks_bot.transforms(joints[mask]))
                indices = compute_manipulability_svd(jacobian)
                new_labels[i] = indices.max()
            else:
                new_labels[i] = 0
        labels[labels == 0] = new_labels

    dof = mdh.shape[0]
    mdh = torch.nn.functional.pad(mdh, (0, 0, 0, 7 - dof), "constant", 0)
    joints = torch.nn.functional.pad(joints, (0, 7 - dof), "constant", 0)

    data = [
        pyarrow.FixedSizeListArray.from_arrays(
            mdh.flatten().unsqueeze(0).expand(labels.shape[0], -1).cpu().numpy().ravel(), 21
        ),
        pyarrow.FixedSizeListArray.from_arrays(poses.numpy().ravel(), poses.shape[1]),
        pyarrow.array(labels.numpy()),
        pyarrow.FixedSizeListArray.from_arrays(joints.numpy().ravel(), 7),
    ]

    schema = pyarrow.schema([
        ("mdh", pyarrow.list_(pyarrow.float32(), 21)),
        ("poses", pyarrow.list_(pyarrow.float32(), poses.shape[1])),
        ("labels", pyarrow.float32()),
        ("joints", pyarrow.list_(pyarrow.float32(), 7)),
    ])

    pattern = rf'{dof}_(\d+)\.parquet'
    matches = [int(m.group(1)) for f in os.listdir(save_folder) if (m := re.match(pattern, f))]
    max_idx = max(matches) if matches else -1
    idx = max_idx + 1

    pyarrow.parquet.write_table(pyarrow.Table.from_arrays(data, schema=schema), save_folder / f"{dof}_{idx}.parquet")
    successful_robots += 1
    pbar.set_description(f"Generated {successful_robots}/{kwargs['num_robots']} workspaces")
