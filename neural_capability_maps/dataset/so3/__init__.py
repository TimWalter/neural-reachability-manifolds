import os
from pathlib import Path

import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped

from scipy.spatial.transform import Rotation

enabled = os.environ.get("SCIPY_ARRAY_API", "").lower() in ("1", "true", "yes")
if not enabled:
    raise RuntimeError(
        "SCIPY_ARRAY_API must be enabled! "
        "Please run: export SCIPY_ARRAY_API=\"true\" before importing this module"
    )



# @jaxtyped(typechecker=beartype)
def distance(x1: Float[Tensor, "*batch 3 3"],
             x2: Float[Tensor, "*batch 3 3"]) -> Float[Tensor, "*batch 1"]:
    """
    Geodesic distance between rotation matrices.

    Args:
        x1: First rotation matrix.
        x2: Second rotation matrix.

    Returns:
        Geodesic distance between x1 and x2.
    """
    r_err = torch.matmul(x1.transpose(-1, -2), x2)
    trace = r_err[..., 0, 0] + r_err[..., 1, 1] + r_err[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    rot_err = torch.arccos(cos_angle)
    return rot_err.unsqueeze(-1)


# @jaxtyped(typechecker=beartype)
def index(orientation: Float[Tensor, "batch 3 3"]) -> Int64[Tensor, "batch"]:
    """
    Get cell index for the given orientation.

    Args:
        orientation: Orientation in SO3

    Returns:
        SO3 cell index
    """
    global _LOOKUP
    if orientation.device != _LOOKUP.device:
        _LOOKUP = _LOOKUP.to(orientation.device)

    rotation_vector = to_index(orientation)

    indices = torch.floor((rotation_vector + torch.pi) / (2 * torch.pi) * _LOOKUP.shape[0]).to(torch.int32)
    indices = torch.clamp(indices, 0, _LOOKUP.shape[0] - 1)  # Against numerical instability

    index = _LOOKUP[indices[:, 0], indices[:, 1], indices[:, 2]]
    return index


# @jaxtyped(typechecker=beartype)
def cell(index: Int64[Tensor, "*batch"]) -> Float[Tensor, "*batch 3 3"]:
    """
    Get cell orientation for the given index.

    Args:
        index: Cell index

    Returns:
        Cell orientation
    """
    global _CELLS
    if index.device != _CELLS.device:
        _CELLS = _CELLS.to(index.device)
    return _CELLS[index]


# @jaxtyped(typechecker=beartype)
def nn(index: Int64[Tensor, "*batch"]) -> Int64[Tensor, "*batch 6"]:
    """
    Get nearest neighbour cell indices for the given index.

    Args:
        index: Cell index

    Returns:
        Nearest neighbour cell indices
    """
    global _NN
    if index.device != _NN.device:
        _NN = _NN.to(index.device)

    return _NN[index]


# @jaxtyped(typechecker=beartype)
def _generate_lookup(n_div: int, cells: Float[Tensor, "n_cells 3 3"]) -> Int64[Tensor, "n_div n_div n_div"]:
    """
    Generate lookup table.

    Args:
        n_div: Number of divisions along each axis.
        cells: Cell centres.

    Returns:
        Lookup table.
    """
    indices = torch.cartesian_prod(*[torch.arange(n_div, device=cells.device)] * 3)
    lookup_centre = Rotation.from_rotvec(((indices + 0.5) / n_div) * 2 * torch.pi - torch.pi).as_matrix()

    nearest_indices_list = []
    n_cells = cells.shape[0]
    num_points = lookup_centre.shape[0]
    # Have to do batches to avoid OOM
    for i in range(0, num_points, 1000):
        batch_centers = lookup_centre[i: i + 1000]
        current_batch_size = batch_centers.shape[0]

        x1 = cells.unsqueeze(0).expand(current_batch_size, n_cells, 3, 3).reshape(-1, 3, 3)
        x2 = batch_centers.unsqueeze(1).expand(current_batch_size, n_cells, 3, 3).reshape(-1, 3, 3)

        distances = distance(x1, x2).squeeze(-1)
        distances = distances.view(current_batch_size, n_cells)

        nearest_idx = torch.argmin(distances, dim=1)
        nearest_indices_list.append(nearest_idx.cpu())

    lookup = torch.cat(nearest_indices_list).view(n_div, n_div, n_div).to(torch.int64)
    return lookup


# @jaxtyped(typechecker=beartype)
def _generate_nn(cells: Float[Tensor, "n_cells 3 3"]) -> Float[Tensor, "n_cells 6"]:
    """
    Generate indices for nearest neighbours.

    Args:
        cells: Cell centres.

    Returns:
        Nearest neighbour indices.
    """
    # Have to do batches to avoid OOM
    nn = []
    for i in range(0, cells.shape[0], 1000):
        batch = cells[i: i + 1000]
        current_batch_size = batch.shape[0]
        x1 = batch.unsqueeze(1).expand(current_batch_size, cells.shape[0], 3, 3)
        x2 = cells.unsqueeze(0).expand(current_batch_size, cells.shape[0], 3, 3)
        distances = distance(x1, x2).squeeze(-1)
        nn += [distances.argsort(dim=-1)[:, 1:7].cpu()]  # Exclude self (first column)
    nn = torch.cat(nn, dim=0)
    return nn.clone().to(torch.int64)


LEVEL = 4

MAX_DISTANCE_BETWEEN_CELLS = {1: 0.6527321338653564, 2: 0.3308008015155792, 3: 0.1654, 4: 0.1390690207}[LEVEL]
_CELLS = torch.load(Path(__file__).parent / f"cells_{LEVEL}.pt", map_location="cpu")  # From RWA
N_CELLS = _CELLS.shape[0]

lookup_path = Path(__file__).parent / f"lookup_{LEVEL}.pt"
if lookup_path.exists():
    _LOOKUP = torch.load(lookup_path, map_location="cpu")
else:
    _LOOKUP = _generate_lookup(128, _CELLS.to("cuda"))
    torch.save(_LOOKUP, lookup_path)

nn_path = Path(__file__).parent / f"nearest_neighbours_{LEVEL}.pt"
if nn_path.exists():
    _NN = torch.load(nn_path, map_location="cpu")
else:
    _NN = _generate_nn(_CELLS.to("cuda"))
    torch.save(_NN, nn_path)


# @jaxtyped(typechecker=beartype)
def random(num_samples: int) -> Float[Tensor, "num_samples 3 3"]:
    """
    Sample random orientations uniformly from SO(3).

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Random orientations.
    """
    quaternion = torch.randn(num_samples, 4)
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
    rotation = Rotation.from_quat(quaternion).as_matrix()

    return rotation


# @jaxtyped(typechecker=beartype)
def to_vector(orientation: Float[Tensor, "*batch 3 3"]) -> Float[Tensor, "*batch 6"]:
    """
    Convert 3x3 rotation matrix to a continuous 6D rotation representation

    Args:
        orientation: Rotation matrix

    Returns:
        6D rotation representation
    """
    return orientation[..., :3, :2].transpose(-1, -2).reshape(*orientation.shape[:-2], 6)


# @jaxtyped(typechecker=beartype)
def from_vector(vec: Float[Tensor, "*batch 6"]) -> Float[Tensor, "*batch 3 3"]:
    """
    Convert continuous 6D rotation representation to 3x3 rotation matrix.

    Args:
        vec: 6D rotation representation

    Returns:
        Rotation matrix
    """
    r1 = vec[..., :3]
    r2 = vec[..., 3:]
    r3 = torch.cross(r1, r2, dim=-1)
    return torch.stack([r1, r2, r3], dim=-1)

@torch.compile
def to_index(orientation: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 3"]:
    """
    Convert 3x3 rotation matrix to rotation vector (axis-angle representation), which we use for indexing the lookup.

    Args:
        orientation: Rotation matrix

    Returns:
        Rotation vector
    """
    return Rotation.from_matrix(orientation, assume_valid=True).as_rotvec()
