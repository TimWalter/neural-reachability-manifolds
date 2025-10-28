import torch
import threading
import pyarrow.parquet
from queue import Queue
from pathlib import Path
import warnings

from data_sampling.se3_cells import N_CELLS

warnings.filterwarnings('ignore', message='.*NumPy array is not writable.*')


class Dataset:
    loader: threading.Thread = None
    data: torch.Tensor = None

    def __init__(self,
                 data_dir: Path,
                 device: torch.device,
                 batch_size: int,
                 only_reachable: bool,
                 minimum_balance: float,
                 prefetch_files: int = 1):
        self.files = list(data_dir.rglob('*.parquet'))
        self.device = device
        self.batch_size = batch_size
        self.only_reachable = only_reachable
        self.minimum_balance = minimum_balance
        self.prefetch_files = prefetch_files


        if self.only_reachable:
            self.num_samples = 0
            for path in self.files:
                self.num_samples += int(pyarrow.parquet.read_metadata(path).metadata[b'reachable_poses'])
        else:
            self.num_samples = len(self.files) * N_CELLS

        # TODO remove
        if batch_size == 10000:
            self.num_samples = self.num_samples // 10
        else:
            self.num_samples = self.num_samples // 10 * 9

        self.batch_queue = Queue(maxsize=prefetch_files * N_CELLS // batch_size)
        self.stop_loading = threading.Event()

    def __iter__(self):
        self.stop_loading.clear()
        self.loader = threading.Thread(target=self._loader_func, daemon=False)
        self.loader.start()

        try:
            for _ in range(len(self)):
                yield self.batch_queue.get(timeout=5)

        finally:
            self.stop_loading.set()

    def __del__(self):
        self.stop_loading.set()

    def __len__(self):
        return self.num_samples // self.batch_size

    def _loader_func(self):
        # Stage 1: Shuffle files
        for file_idx in torch.randperm(len(self.files)).tolist():
            if self.stop_loading.is_set():
                break

            num_reachable = int(pyarrow.parquet.read_metadata(self.files[file_idx]).metadata[b'reachable_poses'])
            num_unreachable = N_CELLS - num_reachable

            if self.batch_size == 10000:
                reachable_indices = torch.randperm(num_reachable // 10)
                unreachable_indices = torch.randperm(num_unreachable // 10)
            else:
                reachable_indices = torch.randperm(num_reachable // 10 * 9) + num_reachable // 10
                unreachable_indices = torch.randperm(num_unreachable // 10 * 9) + num_unreachable // 10

            table = pyarrow.parquet.read_table(self.files[file_idx])
            self.data = torch.cat([
                torch.from_numpy(table['labels'].to_numpy(zero_copy_only=False)).to(self.device).unsqueeze(1),
                torch.from_numpy(table['poses'].combine_chunks().to_numpy_ndarray()).to(self.device),
                torch.from_numpy(table['mdh'].combine_chunks().to_numpy_ndarray()).to(self.device),
            ], dim=1)

            mask = self.data[:, 0] != -1
            reachable_data = self.data[mask][reachable_indices]
            unreachable_data = self.data[~mask][unreachable_indices]
            # TODO return unpadded mdh
            dof = (self.data[0, 10:].view(-1, 3).abs().sum(dim=1) != 0).sum().item()
            if self.only_reachable:
                batch_idx = 0
                while batch_idx < len(reachable_data):
                    batch_data = reachable_data[batch_idx:batch_idx + self.batch_size]
                    self.batch_queue.put((
                        torch.ones(batch_data.shape[0], device=self.device),
                        batch_data[:, 10:].view(self.batch_size, -1, 3)[:, :dof, :],
                        batch_data[:, 1:10],
                        batch_data[:, 0]))
            else:
                weights, reachable_per_batch, unreachable_per_batch = \
                    self._determine_weights_and_ratios(len(reachable_data), len(unreachable_data))

                r_idx = 0
                u_idx = 0
                # TODO: Idea run only for balanced classes, so only until reachable runs out (or 1 at least) with upweighting though on a per robot basis
                while r_idx < len(reachable_indices) or u_idx < len(unreachable_indices):

                    batch_data = torch.cat([
                        unreachable_data[u_idx:u_idx + unreachable_per_batch],
                        reachable_data[r_idx:r_idx + reachable_per_batch]
                    ], dim=0)

                    # Pad with random samples if needed
                    missing_reachable_samples = reachable_per_batch - max(0, reachable_data.shape[0] - r_idx)
                    missing_unreachable_samples = unreachable_per_batch - max(0, unreachable_data.shape[0] - u_idx)
                    if missing_reachable_samples > 0:
                        batch_data = torch.cat([
                            batch_data,
                            reachable_data[torch.randint(0, len(reachable_data), (missing_reachable_samples,))]
                        ])
                    if missing_unreachable_samples > 0:
                        batch_data = torch.cat([
                            batch_data,
                            unreachable_data[torch.randint(0, len(unreachable_data), (missing_unreachable_samples,))]
                        ])
                    labels = batch_data[:, 0]
                    poses = batch_data[:, 1:10]
                    mdh = batch_data[:, 10:].view(self.batch_size, -1, 3)[:, :dof, :]

                    self.batch_queue.put((weights.clone(), mdh, poses, labels))

                    r_idx += reachable_per_batch
                    u_idx += unreachable_per_batch

    def _determine_weights_and_ratios(self, a: int, b: int) -> tuple[torch.Tensor, int, int]:
        ratio = (a if a < b else b) / (a + b)
        weight = 1.0

        if ratio < self.minimum_balance:
            weight = self.minimum_balance / ratio
            ratio = self.minimum_balance

        size_a = int(ratio * self.batch_size) if a < b else self.batch_size - int(ratio * self.batch_size)
        size_b = int(ratio * self.batch_size) if b < a else self.batch_size - int(ratio * self.batch_size)

        if size_a < size_b:
            weights = torch.cat([
                torch.ones(size_a),
                torch.full((size_b,), weight)
            ], dim=0).to(self.device)
        else:
            weights = torch.cat([
                torch.full((size_a,), weight),
                torch.ones(size_b)
            ], dim=0).to(self.device)

        return weights, size_a, size_b
