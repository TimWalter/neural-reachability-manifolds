import math
import torch
from typing import Callable
from beartype import beartype
from jaxtyping import jaxtyped


@jaxtyped(typechecker=beartype)
def estimate_bytes_per_sample(device: torch.device, workload: Callable, args: tuple, probe_size: int) -> int:
    """
    Measures incremental peak bytes/sample for your workload on the given CUDA device.

    Args:
        device: Target CUDA device.
        workload: Function of the form 'workload(batch_size, *args)' that runs your per-batch ops.
        args: Tuple of additional args to pass to 'workload'.
        probe_size: Initial probe batch size.
    Returns:
        Estimated bytes per sample for the workload.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    # Back off the probe size on OOM.
    batch_size = probe_size
    while batch_size >= 1:
        try:
            with torch.no_grad():
                workload(batch_size, *args)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            delta = max(peak - base, 1)
            return math.ceil(delta / batch_size)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise

    raise RuntimeError("Probe failed even with batch_size=1; not enough free memory for a single sample.")


@jaxtyped(typechecker=beartype)
def get_batch_size(device: torch.device, workload: Callable, args: tuple,
                   safety: float = 0.7, probe_size: int = 2048) -> int:
    """
    Chooses a batch size that uses a 'safety' fraction of currently free VRAM. 'workload' is called once for probing only.

    Args:
        device: Target CUDA device.
        workload: Function of the form 'workload(batch_size, *args)' that runs your per-batch ops.
        args: Tuple of additional args to pass to 'workload'.
        safety: Fraction of free memory to use (default 0.7).
        probe_size: Initial probe batch size (default 2048).
    Returns:
        Estimated batch size that fits in 'safety' fraction of free VRAM.
    """
    if device.type != "cuda":
        raise ValueError("CUDA device required for auto-tuning.")

    b_per_sample = estimate_bytes_per_sample(device, workload, args, probe_size)

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
