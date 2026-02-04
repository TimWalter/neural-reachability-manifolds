import re
import argparse
from pathlib import Path

import torch
import zarr
import fasteners
from tqdm import tqdm

import neural_capability_maps.dataset.se3 as se3
from neural_capability_maps.dataset.morphology import sample_morph
from neural_capability_maps.dataset.capability_map import sample_capability_map_analytically, sample_capability_map

CHUNK_SIZE = 100_000  # train: ~2.4MB, val:  ~4.4MB
SHARD_SIZE = CHUNK_SIZE * 1000  # train: ~2.4GB, val:  ~4.4GB

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="train", help="For which set to sample (train/val/test)")
parser.add_argument("--num_robots", type=int, default=10, help="number of robots to generate")
parser.add_argument("--num_samples", type=int, default=1_000_000, help="number of samples to generate per robot")
args = parser.parse_args()

assert args.num_samples % CHUNK_SIZE == 0, f"Only full chunks are supported (chunk size {CHUNK_SIZE})"
assert SHARD_SIZE / args.num_samples   == SHARD_SIZE // args.num_samples, f"One robot must belong to one shard (shard size {SHARD_SIZE})"

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
    existing = [int(re.findall(r'\d+', k)[-1]) for k in root.array_keys() if re.findall(r'\d+', k)]
    file_name = str((max(existing) + 1) if existing else 0)
    print(f"Working in file {file_name}")
    sample_type = "int64" if args.set == "train" else "float32"
    sample_dim = 3 if args.set == "train" else 11
    root.create_array(file_name,
                      shape=(args.num_robots * args.num_samples, sample_dim),
                      dtype=sample_type,
                      chunks=(CHUNK_SIZE, sample_dim),
                      shards=(SHARD_SIZE, sample_dim),
                      compressors=compressor)

morph_file = root[MORPH_FILE_NAME]
file = root[file_name]
file_idx = 0

sample_buffer = torch.empty(0, sample_dim, dtype=torch.int64 if args.set == "train" else torch.float32)

for dof in range(6, 7):
    morphs = sample_morph(args.num_robots, dof, args.set != "train")

    for morph in tqdm(morphs, desc=f"Generating {dof} DOF robots"):
        if args.set == "train":
            cell_indices, labels = sample_capability_map(morph, args.num_samples, minutes=1)

            poses = cell_indices.unsqueeze(1)
            labels = labels.long().unsqueeze(1)
        else:
            poses, labels = sample_capability_map_analytically(morph, args.num_samples)
            poses = se3.to_vector(poses)
            labels = labels.float().unsqueeze(1)

        morph = torch.nn.functional.pad(morph, (0, 0, 0, 8 - morph.shape[0]))
        with lock:
            morph_file.append(morph.unsqueeze(0).numpy(), axis=0)
            morph_id = morph_file.shape[0] - 1

        sample_buffer = torch.cat([sample_buffer, torch.cat([torch.full_like(labels, morph_id), poses, labels], dim=1)])

        if sample_buffer.shape[0] == SHARD_SIZE:
            sample_buffer = sample_buffer[torch.randperm(sample_buffer.shape[0])]

            file[file_idx: file_idx + sample_buffer.shape[0]] = sample_buffer.cpu().numpy()
            file_idx += sample_buffer.shape[0]

            sample_buffer = torch.empty(0, sample_dim, dtype=torch.int64 if args.set == "train" else torch.float32)

sample_buffer = sample_buffer[torch.randperm(sample_buffer.shape[0])]
file[file_idx:file_idx + sample_buffer.shape[0]] = sample_buffer.cpu().numpy()