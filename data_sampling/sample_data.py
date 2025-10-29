import argparse
import os
import re
import torch
import pyarrow.parquet
from estimate_workspace import estimate_workspace
from tqdm import tqdm
from pathlib import Path
from sample_mdh import sample_mdh

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="test", help="what data to generate")
parser.add_argument("--num_robots", type=int, default=1, help="number of robots to generate")
args = parser.parse_args()
kwargs = vars(args)

MODE = kwargs['mode']
NUM_ROB = kwargs['num_robots']
save_folder = Path(__file__).parent.parent / 'data' / MODE
save_folder.mkdir(parents=True, exist_ok=True)

for dof in range(1, 8):
    mdhs = sample_mdh(NUM_ROB * 2, dof=dof)

    successful_robots = 0
    pbar = tqdm(mdhs, desc=f"Generating workspace for robots with {dof} DOF")

    for mdh in pbar:
        if successful_robots >= NUM_ROB:
            break

        poses, labels, joints = estimate_workspace(mdh)

        if labels.sum() == -labels.shape[0]:
            pbar.set_description(f"Skipping robot - no valid configurations found")
            continue

        shuffle = torch.randperm(labels.shape[0])
        poses = poses[shuffle]
        labels = labels[shuffle]
        joints = joints[shuffle]

        mdh = torch.nn.functional.pad(mdh, (0, 0, 0, 7 - dof), "constant", 0)
        joints = torch.nn.functional.pad(joints, (0, 7 - dof), "constant", 0)

        mdh = mdh.flatten().unsqueeze(0).expand(labels.shape[0], -1).contiguous()

        data = {
            "mdh": pyarrow.FixedShapeTensorArray.from_numpy_ndarray(mdh.numpy()),
            "poses": pyarrow.FixedShapeTensorArray.from_numpy_ndarray(poses.numpy()),
            "labels": pyarrow.array(labels.numpy()),
            "joints": pyarrow.FixedShapeTensorArray.from_numpy_ndarray(joints.numpy()),
        }
        table = pyarrow.table(data)
        reachable_poses = int((labels != -1).sum().item())
        table = table.replace_schema_metadata({"reachable_poses": str(reachable_poses)})

        pattern = rf'{dof}_(\d+)\.parquet'
        matches = [int(m.group(1)) for f in os.listdir(save_folder) if (m := re.match(pattern, f))]
        max_idx = max(matches) if matches else -1
        idx = max_idx + 1

        pyarrow.parquet.write_table(table, save_folder / f"{dof}_{idx}.parquet", compression="BROTLI")
        successful_robots += 1
        pbar.set_description(
            f"Generated {successful_robots}/{NUM_ROB} workspaces for robots with {dof} DOF, reachable poses {reachable_poses}")
