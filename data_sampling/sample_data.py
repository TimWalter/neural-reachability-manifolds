import argparse
from pathlib import Path

import torch
import zarr
from tqdm import tqdm
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float

from data_sampling.representations import homogeneous_to_vector
from sample_morph import sample_morph
from estimate_workspace import estimate_workspace, estimate_workspace_analytically


compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="train", help="what data to generate")
parser.add_argument("--num_robots", type=int, default=1, help="number of robots to generate")
parser.add_argument("--num_samples", type=int, default=10_000_000, help="number of samples to generate per robot")
parser.add_argument("--chunk_size", type=int, default=1_000, help="minimum read size for zarr arrays")
parser.add_argument("--shard_size", type=int, default=1, help="number of robots per shard in zarr arrays")
args = parser.parse_args()

NUM_ROB = args.num_robots
NUM_SAMPLES = args.num_samples
CHUNK_SIZE = args.chunk_size
SHARD_SIZE = args.shard_size
ANALYTICAL = args.folder != "train"

save_folder = Path(__file__).parent.parent / 'data' / args.folder
save_folder.mkdir(parents=True, exist_ok=True)
root = zarr.open(save_folder, mode="a")

morph_store_name = "morphologies"
if morph_store_name not in root:
    root.create_array(
        morph_store_name,
        shape=(0, 8, 3),
        dtype="float32",
        compressors=compressor,
        overwrite=False,
    )
morph_store = root[morph_store_name]


# @jaxtyped(typechecker=beartype)
def save_shard(samples: Float[Tensor, "NUM_SAMPLES_x_SHARD_SIZE 3"]):
    if ANALYTICAL:
        samples = samples[torch.randperm(samples.shape[0])]

    existing = [int(k) for k in root.array_keys() if k.isdigit()]
    shard_idx = max(existing) + 1 if existing else 0
    shard_name = str(shard_idx)
    root.create_array(shard_name, data=samples.numpy(),
                      chunks=(CHUNK_SIZE, samples.shape[1]),
                      shards=(SHARD_SIZE * NUM_SAMPLES, samples.shape[1]),
                      compressors=compressor)

# TODO remove, for now only to get the same train and val morphs
torch.manual_seed(0)

if ANALYTICAL:
    aggregated_samples = torch.empty((0, 11), dtype=torch.float32)
else:
    aggregated_samples = torch.empty((0, 3), dtype=torch.int64)

for dof in range(6, 7):
    morphs = sample_morph(NUM_ROB, dof, True) # TODO True -> ANALYTICAL

    for morph in tqdm(morphs, desc=f"Generating {dof} DOF robots"):
        if ANALYTICAL:
            labels, poses = estimate_workspace_analytically(morph, NUM_SAMPLES)
            poses = homogeneous_to_vector(poses)
        else:
            labels, cell_indices = estimate_workspace(morph)

            subset = torch.randperm(labels.shape[0])[:NUM_SAMPLES]
            labels = labels[subset]
            cell_indices = cell_indices[subset]
            subsamples = (labels.unsqueeze(1), cell_indices.unsqueeze(1))

        morph = torch.nn.functional.pad(morph, (0, 0, 0, 8 - morph.shape[0]))
        morph_store.append(morph.unsqueeze(0).numpy(), axis=0)
        morph_id = morph_store.shape[0] - 1

        # TODO balance samples?

        samples = torch.cat([*subsamples, torch.full_like(subsamples[0], morph_id)], dim=1)

        aggregated_samples = torch.cat([aggregated_samples, samples], dim=0)

        if aggregated_samples.shape[0] == SHARD_SIZE * NUM_SAMPLES:
            save_shard(aggregated_samples)
            if ANALYTICAL:
                aggregated_samples = torch.empty((0, 11), dtype=torch.float32)
            else:
                aggregated_samples = torch.empty((0, 3), dtype=torch.int64)

if aggregated_samples.shape[0] > 0:
    save_shard(aggregated_samples)
