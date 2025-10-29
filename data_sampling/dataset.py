import torch
import threading
import pyarrow.parquet
from queue import Queue
from pathlib import Path
import warnings

from data_sampling.estimate_workspace import NUM_SAMPLES

warnings.filterwarnings('ignore', message='.*NumPy array is not writable.*')


class Dataset:
    loader: threading.Thread = None
    data: torch.Tensor = None

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 only_reachable: bool,
                 prefetch_files: int = 1):
        self.files = list(data_dir.rglob('*.parquet'))
        self.batch_size = batch_size
        self.only_reachable = only_reachable
        self.prefetch_files = prefetch_files

        self.num_samples = len(self.files) * NUM_SAMPLES
        if only_reachable:
            self.num_samples = 0
            for path in self.files:
                self.num_samples += int(pyarrow.parquet.read_metadata(path).metadata[b'reachable_poses'])


        # TODO remove
        if batch_size == 10000:
            self.num_samples = self.num_samples // 10
        else:
            self.num_samples = self.num_samples // 10 * 9

        self.batch_queue = Queue(maxsize=prefetch_files * NUM_SAMPLES // batch_size)
        self.stop_loading = threading.Event()

    def __iter__(self):
        self.stop_loading.clear()
        self.loader = threading.Thread(target=self._loader_func, daemon=False)
        self.loader.start()

        try:
            for _ in range(len(self)):
                yield self.batch_queue.get(timeout=10)

        finally:
            self.stop_loading.set()

    def __del__(self):
        self.stop_loading.set()

    def __len__(self):
        return self.num_samples // self.batch_size

    def _loader_func(self):
        # Stage 1: Shuffle files
        for file_idx in torch.randperm(len(self.files)).tolist():

            if self.batch_size == 10000:
                shuffle = torch.randperm(NUM_SAMPLES // 10)
            else:
                shuffle = torch.randperm(NUM_SAMPLES // 10 * 9) + NUM_SAMPLES // 10

            table = pyarrow.parquet.read_table(self.files[file_idx])
            self.data = torch.cat([
                torch.from_numpy(table['labels'].to_numpy(zero_copy_only=False)).unsqueeze(1),
                torch.from_numpy(table['poses'].combine_chunks().to_numpy_ndarray()),
                torch.from_numpy(table['mdh'].combine_chunks().to_numpy_ndarray()),
            ], dim=1)
            if self.only_reachable:
                self.data = self.data[self.data[:, 0] != -1]

            # Stage 2: Shuffle samples within file
            self.data = self.data[shuffle]
            dof = (self.data[0, 10:].view(-1, 3).abs().sum(dim=1) != 0).sum().item()

            for batch_start in range(0, NUM_SAMPLES, self.batch_size):
                batch_data = self.data[batch_start:batch_start + self.batch_size]
                if batch_data.size(0) < self.batch_size:
                    break

                labels = batch_data[:, 0]
                poses = batch_data[:, 1:10]
                mdh = batch_data[:, 10:10+dof*3].view(-1, dof, 3)

                self.batch_queue.put((mdh, poses, labels))
                if self.stop_loading.is_set():
                    break
