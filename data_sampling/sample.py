import re
import argparse
from pathlib import Path

import torch
import zarr
import fasteners
from tqdm import tqdm

from data_sampling.sample_morph import sample_morph
from data_sampling.representations import homogeneous_to_vector
from data_sampling.estimate_workspace import estimate_workspace, estimate_workspace_analytically

CHUNK_SIZE = 100_000 # ~100MB
SHARD_SIZE = CHUNK_SIZE * 100 # ~10GB

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="val", help="For which set to sample (train/val/test)")
parser.add_argument("--num_robots", type=int, default=1, help="number of robots to generate")
parser.add_argument("--num_samples", type=int, default=100_000, help="number of samples to generate per robot")
args = parser.parse_args()
assert args.num_samples * args.num_robots % CHUNK_SIZE == 0, f"Only full chunks are supported (chunk size {CHUNK_SIZE})"

SAFE_FOLDER = Path(__file__).parent.parent / 'data' / args.set
MORPH_FILE_NAME = "morphologies"
lock = fasteners.InterProcessLock(SAFE_FOLDER.parent / f'{args.set}_lock.file')
compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

with lock:
    SAFE_FOLDER.mkdir(parents=True, exist_ok=True)
root = zarr.open(SAFE_FOLDER, mode="a")

with lock:
    if MORPH_FILE_NAME not in root:
        root.create_array(
            MORPH_FILE_NAME,
            shape=(0, 8, 3),
            dtype="float32",
            compressors=compressor,
            overwrite=False,
        )
    existing = [re.findall(r'\d+', k)[-1] for k in root.array_keys() if re.findall(r'\d+', k)]
    file_name = str((max(existing) + 1) if existing else 0)
    print(f"Working in file {file_name}")
    sample_type = "int64" if args.set == "train" else "float32"
    sample_dim = 3 if args.set == "train" else 11
    root.create_array(file_name,
                      shape=(args.num_robots * args.num_samples,  sample_dim),
                      dtype=sample_type,
                      chunks=(CHUNK_SIZE, sample_dim),
                      shards=(SHARD_SIZE, sample_dim),
                      compressors=compressor)

morph_file = root[MORPH_FILE_NAME]
file = root[file_name]
file_idx = 0

torch.manual_seed(1) # TODO remove, for now only to get the same train morphs
for dof in range(6, 7):
    morphs = sample_morph(args.num_robots, dof, True) # TODO True -> args.set != "train"

    for morph in tqdm(morphs, desc=f"Generating {dof} DOF robots"):
        if args.set == "train":
            cell_indices, labels = estimate_workspace(morph)

            r_subset = torch.randperm(labels.shape[0]//2)[:args.num_samples//2]
            u_subset = r_subset + labels.shape[0]//2

            poses = torch.cat([cell_indices[r_subset], cell_indices[u_subset]], dim=0).unsqueeze(1)
            labels = torch.cat([labels[r_subset], labels[u_subset]], dim=0).long().unsqueeze(1)
        else:
            poses, labels = estimate_workspace_analytically(morph, args.num_samples)
            poses = homogeneous_to_vector(poses)
            labels = labels.unsqueeze(1).float()

        morph = torch.nn.functional.pad(morph, (0, 0, 0, 8 - morph.shape[0]))
        with lock:
            morph_file.append(morph.unsqueeze(0).numpy(), axis=0)
            morph_id = morph_file.shape[0] - 1

        samples = torch.cat([torch.full_like(labels, morph_id), poses, labels], dim=1)
        samples = samples[torch.randperm(samples.shape[0])]

        file[file_idx : file_idx + samples.shape[0]] = samples.cpu().numpy()