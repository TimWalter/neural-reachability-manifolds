import torch

import data_sampling.so3 as so3

from scipy.spatial.transform import Rotation


def test_distance():
    x1 = torch.eye(3).repeat(3, 1, 1)
    x2 = torch.stack([
        torch.tensor([[1, 0, 0],
                      [0, -1, 0],
                     [0, 0, -1]]),
        torch.tensor([[-1, 0, 0],
                      [0, 1, 0],
                     [0, 0, -1]]),
        torch.tensor([[-1, 0, 0],
                      [0, -1, 0],
                     [0, 0, 1]])
    ],dim=0).float()

    d = so3.distance(x1, x2)

    assert d.shape == (3, 1)
    assert torch.allclose(d, torch.tensor([torch.pi]))

def test_max_distance_between_cells():
    cells = so3.cell(torch.arange(so3.N_CELLS))
    # Have to do batches to avoid OOM
    distances = []
    for i in range(0, so3.N_CELLS, 1000):
        distances += [
            so3.distance(
                cells[i:i + 1000].unsqueeze(1).expand(min(1000, so3.N_CELLS - i), so3.N_CELLS, 3, 3),
                cells.unsqueeze(0).expand(min(1000, so3.N_CELLS - i), so3.N_CELLS, 3, 3)
            )
        ]
    distances = torch.cat(distances, dim=0).squeeze(-1)
    distances = distances.sort(dim=1).values  # the smallest distance is with themselves
    assert torch.allclose(distances[:, 1:7].max(), torch.tensor([so3.MAX_DISTANCE_BETWEEN_CELLS]))

def test_index_cell_consistency():
    test_indices = torch.tensor([0, 1, 2, 3, 4])

    cells = so3.cell(test_indices)
    retrieved_indices = so3.index(cells)

    assert torch.equal(test_indices, retrieved_indices)

def test_lookup():
    quaternions = torch.randn(100_000, 4)
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    rotation = Rotation.from_quat(quaternions).as_matrix()

    max_rot_distances = so3.distance(so3.cell(so3.index(rotation)), rotation).max()

    assert max_rot_distances < so3.MAX_DISTANCE_BETWEEN_CELLS

def test_nn():
    test_index = torch.tensor([10])
    nn_indices = so3.nn(test_index)

    centre_cell = so3.cell(test_index)
    nn_cells = so3.cell(nn_indices)

    distances = so3.distance(centre_cell.unsqueeze(1).repeat(1, 6, 1, 1), nn_cells)

    assert torch.all(distances.abs() <= torch.tensor([so3.MAX_DISTANCE_BETWEEN_CELLS]))