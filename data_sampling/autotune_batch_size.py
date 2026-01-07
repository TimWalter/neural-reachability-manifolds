import math
import torch
from typing import Callable
from beartype import beartype
from jaxtyping import jaxtyped


#@jaxtyped(typechecker=beartype)
def estimate_bytes(device: torch.device, workload: Callable, args: list) -> int:
    """
    Measures incremental peak bytes/sample for your workload on the given CUDA device.

    Args:
        device: Target CUDA device.
        workload: Function that runs your per-batch ops.
        args: List of additional args to pass to 'workload'.
    Returns:
        Estimated bytes per sample for the workload.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        workload(*args)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    delta = max(peak - base, 1)
    return delta


#@jaxtyped(typechecker=beartype)
def get_batch_size(device: torch.device, workload: Callable, probe_size: int, args: list, safety: float = 0.7) -> int:
    """
    Chooses a batch size that uses a 'safety' fraction of currently free VRAM. 'workload' is called once for probing only.

    Args:
        device: Target CUDA device.
        workload: Function that runs your per-batch ops.
        probe_size: Initial probe batch size (default 2048).
        args: List of additional args to pass to 'workload'.
        safety: Fraction of free memory to use (default 0.7).
    Returns:
        Estimated batch size that fits in 'safety' fraction of free VRAM.
    """
    if device.type != "cuda":
        raise ValueError("CUDA device required for auto-tuning.")

    b_per_sample = math.ceil(estimate_bytes(device, workload, args) / probe_size)

    with torch.cuda.device(device):
        free_bytes, _ = torch.cuda.mem_get_info()

    usable = int(free_bytes * float(safety))
    batch_size = max(1, usable // max(b_per_sample, 1))

    print(
        f"[auto-batch] est. bytes/sample: {b_per_sample} "
        f"({b_per_sample / 1024:.2f} KiB), free: {free_bytes / 1024 ** 3:.2f} GiB, "
        f"safety: {safety}, batch_size: {batch_size}"
    )
    return batch_size
