import torch
import threading
import pyarrow.parquet
from queue import Queue
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', message='.*NumPy array is not writable.*')


class Dummyset:

    def __init__(self,
                 data_dir: Path,
                 device: torch.device,
                 batch_size: int,
                 only_reachable: bool,
                 minimum_balance: float,
                 prefetch_files: int = 5):
        self.device = device
        self.batch_size = batch_size
        self.num_samples = 1e7

    def __len__(self):
        return int(self.num_samples // self.batch_size)*7

    def __iter__(self):
        weights = torch.ones(self.batch_size).to(self.device)
        mdh = torch.randn(self.batch_size, 8, 3).to(self.device)
        poses = torch.randn(self.batch_size, 9).to(self.device)
        labels = torch.rand(self.batch_size).to(self.device) * 2 - 1

        for _ in range(int(self.num_samples // self.batch_size)*7):
            yield weights.detach(), mdh.detach(), poses.detach(), labels.detach()



class Dataset:
    loader: threading.Thread = None
    mdh : torch.Tensor = None
    poses : torch.Tensor = None
    labels : torch.Tensor = None
    reachable_mdh: torch.Tensor = None
    reachable_poses: torch.Tensor = None
    reachable_labels: torch.Tensor = None
    unreachable_mdh: torch.Tensor = None
    unreachable_poses: torch.Tensor = None
    unreachable_labels: torch.Tensor = None

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

        self.batch_queue = Queue(maxsize=
                                 prefetch_files *
                                 torch.load(Path(__file__).parent / "r3_cells.pt", map_location="cpu").shape[0] *
                                 torch.load(Path(__file__).parent / "so3_cells.pt", map_location="cpu").shape[0] //
                                 batch_size)
        self.stop_loading = threading.Event()

        if self.only_reachable:
            self.num_samples = 0
            for path in self.files:
                self.num_samples += int(pyarrow.parquet.read_metadata(path).metadata[b'reachable_poses'])
        else:
            nr_cells = torch.load(Path(__file__).parent / "r3_cells.pt", map_location="cpu").shape[0] * \
                       torch.load(Path(__file__).parent / "so3_cells.pt", map_location="cpu").shape[0]
            self.num_samples = len(self.files) * nr_cells

        if batch_size == 10000:
            self.num_samples = self.num_samples // 10
        else:
            self.num_samples = self.num_samples // 10* 9

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

            table = pyarrow.parquet.read_table(self.files[file_idx])
            self.mdh = torch.from_numpy(table['mdh'].combine_chunks().to_numpy_ndarray()).to(self.device)
            self.poses = torch.from_numpy(table['poses'].combine_chunks().to_numpy_ndarray()).to(self.device)
            self.labels = torch.from_numpy(table['labels'].to_numpy(zero_copy_only=False)).to(self.device)

            mask = self.labels != -1
            self.reachable_mdh = self.mdh[mask]
            self.reachable_poses = self.poses[mask]
            self.reachable_labels = self.labels[mask]
            self.unreachable_mdh = self.mdh[~mask]
            self.unreachable_poses = self.poses[~mask]
            self.unreachable_labels = self.labels[~mask]

            rechable_cutoff = len(self.reachable_mdh)//10
            unreachable_cutoff = len(self.unreachable_mdh)//10
            if self.batch_size == 10000:
                self.reachable_mdh = self.reachable_mdh[:rechable_cutoff]
                self.reachable_poses = self.reachable_poses[:rechable_cutoff]
                self.reachable_labels = self.reachable_labels[:rechable_cutoff]
                self.unreachable_mdh = self.unreachable_mdh[:unreachable_cutoff]
                self.unreachable_poses = self.unreachable_poses[:unreachable_cutoff]
                self.unreachable_labels = self.unreachable_labels[:unreachable_cutoff]
            else:
                #all other
                self.reachable_mdh = self.reachable_mdh[rechable_cutoff:]
                self.reachable_poses = self.reachable_poses[rechable_cutoff:]
                self.reachable_labels = self.reachable_labels[rechable_cutoff:]
                self.unreachable_mdh = self.unreachable_mdh[unreachable_cutoff:]
                self.unreachable_poses = self.unreachable_poses[unreachable_cutoff:]
                self.unreachable_labels = self.unreachable_labels[unreachable_cutoff:]


            reachable_indices = torch.randperm(self.reachable_labels.shape[0])
            unreachable_indices = torch.randperm(self.unreachable_labels.shape[0])

            if self.only_reachable:
                for i in range(0, len(reachable_indices), self.batch_size):
                    batch_idx = reachable_indices[i:i + self.batch_size]
                    self.batch_queue.put((
                        torch.ones(self.batch_size),
                        self.reachable_mdh[batch_idx].view(self.batch_size, -1, 3),
                        self.reachable_poses[batch_idx],
                        self.reachable_labels[batch_idx]
                    ))
            else:
                weight, reachable_per_batch, unreachable_per_batch = \
                    self._determine_weights_and_ratios(len(reachable_indices), len(unreachable_indices))

                if reachable_per_batch < unreachable_per_batch:
                    weights = torch.cat([
                        torch.ones(reachable_per_batch),
                        torch.full((unreachable_per_batch,), weight)
                    ], dim=0).to(self.device)
                else:
                    weights = torch.cat([
                        torch.full((reachable_per_batch,), weight),
                        torch.ones(unreachable_per_batch)
                    ], dim=0).to(self.device)

                r_idx = 0
                u_idx = 0

                while r_idx < len(reachable_indices) or u_idx < len(unreachable_indices): # TODO: Idea run only for balanced classes, so only until reachable runs out (or 1 at least) with upweighting though on a per robot basis
                    batch_r_idx = reachable_indices[r_idx:r_idx + reachable_per_batch]
                    batch_u_idx = unreachable_indices[u_idx:u_idx + unreachable_per_batch]

                    # Pad with random samples if needed
                    if len(batch_r_idx) < reachable_per_batch:
                        batch_r_idx = torch.cat([
                            batch_r_idx,
                            reachable_indices[
                                torch.randint(0, len(reachable_indices), (reachable_per_batch - len(batch_r_idx),))]
                        ])
                    if len(batch_u_idx) < unreachable_per_batch:
                        batch_u_idx = torch.cat([
                            batch_u_idx,
                            unreachable_indices[
                                torch.randint(0, len(unreachable_indices), (unreachable_per_batch - len(batch_u_idx),))]
                        ])

                    self.batch_queue.put((
                        weights.clone(),
                        torch.cat([
                            self.reachable_mdh[batch_r_idx],
                            self.unreachable_mdh[batch_u_idx]
                        ]).view(self.batch_size, -1, 3),
                        torch.cat([
                            self.reachable_poses[batch_r_idx],
                            self.unreachable_poses[batch_u_idx]
                        ]),
                        torch.cat([
                            self.reachable_labels[batch_r_idx],
                            self.unreachable_labels[batch_u_idx]
                        ])
                    ))

                    r_idx += reachable_per_batch
                    u_idx += unreachable_per_batch

    def _determine_weights_and_ratios(self, a: int, b: int) -> tuple[float, int, int]:
        ratio = (a if a < b else b) / (a + b)
        weight = 1.0

        if ratio < self.minimum_balance:
            weight = self.minimum_balance / ratio
            ratio = self.minimum_balance

        size_a = int(ratio * self.batch_size) if a < b else self.batch_size - int(ratio * self.batch_size)
        size_b = int(ratio * self.batch_size) if b < a else self.batch_size - int(ratio * self.batch_size)

        return weight, size_a, size_b
