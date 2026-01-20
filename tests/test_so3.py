import torch

import neural_capability_maps.dataset.so3 as so3

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
    cell_indices = torch.arange(so3.N_CELLS)
    cells = so3.cell(cell_indices)
    nn = so3.cell(so3.nn(cell_indices))
    distances = so3.distance(cells.unsqueeze(1).expand(cells.shape[0], nn.shape[1], 3, 3), nn)
    max_distance = distances.max()
    assert torch.allclose(max_distance, torch.tensor([so3.MAX_DISTANCE_BETWEEN_CELLS]))

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

def test_vector():
    rot_mat = so3.random(100)

    cont = so3.to_vector(rot_mat)
    rot_mat_reconstructed = so3.from_vector(cont)

    assert torch.allclose(rot_mat, rot_mat_reconstructed)
