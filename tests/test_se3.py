import torch

import neural_capability_maps.dataset.se3 as se3

from scipy.spatial.transform import Rotation


def test_distance():
    x1 = torch.eye(4).repeat(3, 1, 1)
    x2 = torch.eye(4).repeat(3, 1, 1)
    x2[:, :3, :3] = torch.stack([
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
    x2[:, :3, 3] = torch.ones(3, 3)

    d = se3.distance(x1, x2)

    assert d.shape == (3, 1)
    assert torch.allclose(d, torch.sqrt(torch.tensor([14]))/4)


def test_max_distance_between_cells():
    cells = se3.cell(torch.arange(10_000)) # Cannot check all (way too many)
    # Have to do batches to avoid OOM
    distances = []
    for i in range(0, 10_000, 1000):
        distances += [
            se3.distance(
                cells[i:i + 1000].unsqueeze(1).expand(min(1000, 10_000 - i), 10_000, 4, 4),
                cells.unsqueeze(0).expand(min(1000, 10_000 - i), 10_000, 4, 4)
            )
        ]
    distances = torch.cat(distances, dim=0).squeeze(-1)
    distances = distances.sort(dim=1).values  # the smallest distance is with themselves
    assert (distances[:, 1].max()< torch.tensor([se3.MAX_DISTANCE_BETWEEN_CELLS])).all() # < , cause we cant check all


def test_index_cell_consistency():
    test_indices = torch.tensor([0, 1, 2, 3, 4])

    cells = se3.cell(test_indices)
    retrieved_indices = se3.index(cells)

    assert torch.equal(test_indices, retrieved_indices)


def test_lookup():
    translation = torch.randn(100_000, 3)
    translation /= torch.norm(translation, dim=1, keepdim=True)
    translation *= torch.pow(torch.rand(100_000, 1), 1.0 / 3)

    quaternions = torch.randn(100_000, 4)
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    rotation = Rotation.from_quat(quaternions).as_matrix()

    pose = torch.eye(4).unsqueeze(0).repeat(100_000, 1, 1)
    pose[:, :3, 3] = translation
    pose[:, :3, :3] = rotation

    max_trans_distances = se3.distance(se3.cell(se3.index(pose)), pose).max()

    assert max_trans_distances < se3.MAX_DISTANCE_BETWEEN_CELLS


def test_nn():
    test_index = torch.tensor([10])
    nn_indices = se3.nn(test_index)

    centre_cell = se3.cell(test_index)
    nn_cells = se3.cell(nn_indices)

    distances = se3.distance(centre_cell.unsqueeze(1).repeat(1, 12, 1, 1), nn_cells)

    assert (distances.abs() < se3.MAX_DISTANCE_BETWEEN_CELLS).all()

def test_vector():
    homogeneous = se3.random(100)

    vec = se3.to_vector(homogeneous)
    homogeneous_reconstructed = se3.from_vector(vec)

    assert torch.allclose(homogeneous, homogeneous_reconstructed)