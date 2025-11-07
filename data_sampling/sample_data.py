import argparse
from pathlib import Path

import torch
import zarr
from tqdm import tqdm
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float

from sample_morph import sample_morph
from estimate_workspace import estimate_workspace

compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="train", help="what data to generate")
parser.add_argument("--num_robots", type=int, default=2, help="number of robots to generate")
parser.add_argument("--num_samples", type=int, default=10_000, help="number of samples to generate per robot")
parser.add_argument("--chunk_size", type=int, default=1_000, help="minimum read size for zarr arrays")
parser.add_argument("--shard_size", type=int, default=2, help="number of robots per shard in zarr arrays")
args = parser.parse_args()

NUM_ROB = args.num_robots
NUM_SAMPLES = args.num_samples
CHUNK_SIZE = args.chunk_size
SHARD_SIZE = args.shard_size

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

@jaxtyped(typechecker=beartype)
def save_shard(samples: Float[Tensor, "NUM_SAMPLES_x_SHARD_SIZE 3"]):
    existing = [int(k) for k in root.array_keys() if k.isdigit()]
    shard_idx = max(existing) + 1 if existing else 0
    shard_name = str(shard_idx)
    root.create_array(shard_name, data=samples.numpy(),
                      chunks=(CHUNK_SIZE, samples.shape[1]),
                      shards=(SHARD_SIZE*NUM_SAMPLES, samples.shape[1]),
                      compressors=compressor)


aggregated_samples = torch.empty((0, 3), dtype=torch.float32)
for dof in range(7, 8):
    morphs = sample_morph(NUM_ROB * 2, dof=dof)
    successful_robots = 0

    for morph in tqdm(morphs, desc=f"Generating {dof} DOF robots", total=NUM_ROB):
        labels = estimate_workspace(morph)
        if (labels == -1).all():
            continue
        morph = torch.nn.functional.pad(morph, (0, 0, 0, 7 - dof))
        morph_store.append(morph.unsqueeze(0).numpy(), axis=0)
        morph_id = morph_store.shape[0] - 1

        # Balance dataset with replacement
        mask = labels != -1
        sampled_indices = torch.empty((0,), dtype=torch.long)
        for current_mask in [mask, ~mask]:
            indices = torch.arange(labels.shape[0])[current_mask]
            num_indices = indices.shape[0]
            missing_indices = max(0, NUM_SAMPLES // 2 - num_indices)
            sampled_indices = torch.cat([
                sampled_indices,
                indices[torch.randperm(num_indices)][:NUM_SAMPLES // 2],
                indices[torch.randint(0, num_indices, (missing_indices,))]
            ], dim=0)
        labels = labels[sampled_indices]

        samples = torch.stack([labels, sampled_indices, torch.full_like(labels, morph_id)], dim=1)
        samples = samples[torch.randperm(samples.shape[0])]

        aggregated_samples = torch.cat([aggregated_samples, samples], dim=0)

        if aggregated_samples.shape[0] == SHARD_SIZE * NUM_SAMPLES:
            save_shard(aggregated_samples)
            aggregated_samples = torch.empty((0, 3), dtype=torch.float32)

        successful_robots += 1
        if successful_robots >= NUM_ROB:
            break

if aggregated_samples.shape[0] > 0:
    save_shard(aggregated_samples)