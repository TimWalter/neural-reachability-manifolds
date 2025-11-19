import math
import zarr
import torch
import random
import bisect
from pathlib import Path
from data_sampling.se3_cells import SE3_CELLS
from data_sampling.orientation_representations import rotation_matrix_to_ml


class Dataset:
    def __init__(self, store_path: Path, batch_size: int, shuffle: bool, training: bool):
        self.root = zarr.open(str(store_path), mode="r")
        self.keys = sorted([k for k in self.root.array_keys() if k.isdigit()], key=int)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training

        self.batches_per_shard = [math.ceil(self.root[k].shape[0] / batch_size) for k in self.keys]
        self.offsets = torch.cumsum(torch.tensor(self.batches_per_shard), dim=0).tolist()
        self.num_batches = self.offsets[-1]

        self.morphologies = torch.from_numpy(self.root["morphologies"][:].astype("float32", copy=False))
        self.poses = torch.cat([SE3_CELLS[:, :3, 3], rotation_matrix_to_ml(SE3_CELLS[:, :3, :3])], dim=-1).contiguous()

    def __len__(self) -> int:
        return self.num_batches

    def get_key(self, batch_idx: int) -> tuple[str, int, int]:
        shard_idx = bisect.bisect_right(self.offsets, batch_idx)
        local_batch_idx = batch_idx - (0 if shard_idx == 0 else self.offsets[shard_idx - 1])

        return self.keys[shard_idx], local_batch_idx * self.batch_size, (local_batch_idx + 1) * self.batch_size

    def __getitem__(self, batch_idx: int):
        key, start, end = self.get_key(batch_idx)
        batch = torch.from_numpy(self.root[key][start:end])

        labels = (batch[:, 0:1] != -1).float()
        poses = self.poses[batch[:, 1].long()]
        morph = self.morphologies[batch[:, 2].long()]

        if self.training:
            dof = (morph[0].abs().sum(dim=1) != 0).sum().item()
            morph = morph[:, :dof, :]
        else:
            mask = (morph.abs().sum(dim=2) != 0)
            dofs = mask.sum(dim=1)
            flat = morph[mask]
            split_sizes = dofs.tolist()
            chunks = list(torch.split(flat, split_sizes))
            morph = torch.nested.nested_tensor(chunks, layout=torch.jagged)

        return morph.pin_memory(), poses.pin_memory(), labels.pin_memory()

    def __iter__(self):
        all_batches = list(range(self.num_batches))
        if self.shuffle:
            random.shuffle(all_batches)

        for global_batch_idx in all_batches:
            yield self[global_batch_idx]

    def get_random_batch(self):
        batch_idx = torch.randint(0, self.num_batches, (1,)).item()
        return self[batch_idx]


class SingleDataset(Dataset):
    def __init__(self, store_path: Path, batch_size: int, shuffle: bool, training: bool):
        super().__init__(store_path, batch_size, shuffle, training)
        self.keys = self.keys[-1:]

        self.batches_per_shard = [0]
        data = self.root[self.keys[0]]
        morph_id = data[0, -1]
        for i in range(0, data.shape[0], 1000):
            if data[i, -1] == morph_id:
                self.batches_per_shard[0] += 1
            else:
                break
        self.offsets = torch.cumsum(torch.tensor(self.batches_per_shard), dim=0).tolist()
        self.num_batches = self.offsets[-1]
