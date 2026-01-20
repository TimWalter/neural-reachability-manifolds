import torch

import neural_capability_maps.dataset.r3 as r3


def test_distance():
    x1 = torch.zeros(10, 3)
    x2 = torch.ones(10, 3)

    d = r3.distance(x1, x2)

    assert d.shape == (10, 1)
    assert torch.allclose(d, torch.sqrt(torch.tensor([3])))


def test_max_distance_between_cells():
    cells = r3.cell(torch.arange(r3.N_CELLS))
    # Have to do batches to avoid OOM
    distances = []
    for i in range(0, r3.N_CELLS, 1000):
        distances += [
            r3.distance(
                cells[i:i + 1000].unsqueeze(1).expand(min(1000, r3.N_CELLS - i), r3.N_CELLS, 3),
                cells.unsqueeze(0).expand(min(1000, r3.N_CELLS - i), r3.N_CELLS, 3)
            )
        ]
    distances = torch.cat(distances, dim=0).squeeze(-1)
    distances = distances.sort(dim=1).values  # the smallest distance is with themselves
    assert torch.allclose(distances[:, 1].max(), torch.tensor([r3.MAX_DISTANCE_BETWEEN_CELLS]))


def test_index():
    assert r3.index(torch.tensor([[-1.0, -1.0, -1.0]]))[0] == 0
    assert r3.index(torch.tensor([[1.0, 1.0, 1.0]]))[0] == r3.N_DIV ** 3 - 1
    assert r3.index(torch.tensor([[0.0, 0.0, 0.0]]))[0] == \
           r3.N_DIV // 2 * r3.N_DIV ** 2 + r3.N_DIV // 2 * r3.N_DIV + r3.N_DIV // 2
    assert r3.index(torch.tensor([[1.0, -1.0, -1.0]]))[0] == (r3.N_DIV - 1)
    assert r3.index(torch.tensor([[-1.0, 1.0, -1.0]]))[0] == (r3.N_DIV - 1) * r3.N_DIV
    assert r3.index(torch.tensor([[-1.0, -1.0, 1.0]]))[0] == (r3.N_DIV - 1) * r3.N_DIV ** 2
    assert r3.index(torch.tensor([[1.0, 1.0, -1.0]]))[0] == (r3.N_DIV - 1) + (r3.N_DIV - 1) * r3.N_DIV
    assert r3.index(torch.tensor([[1.0, -1.0, 1.0]]))[0] == (r3.N_DIV - 1) + (r3.N_DIV - 1) * r3.N_DIV ** 2
    assert r3.index(torch.tensor([[-1.0, 1.0, 1.0]]))[0] == (r3.N_DIV - 1) * r3.N_DIV + (r3.N_DIV - 1) * r3.N_DIV ** 2


def test_cell():
    assert torch.allclose(r3.cell(torch.tensor([0])), torch.tensor([(0.5 / r3.N_DIV) * 2.0 - 1.0]))
    assert torch.allclose(r3.cell(torch.tensor([r3.N_DIV ** 3 - 1])),
                          torch.tensor([((r3.N_DIV - 1 + 0.5) / r3.N_DIV) * 2.0 - 1.0]))


def test_index_cell_consistency():
    test_indices = torch.tensor([
        0,
        1,
        r3.N_DIV ** 2 + r3.N_DIV + 2,
        (r3.N_DIV // 2) * r3.N_DIV ** 2 + (r3.N_DIV // 2) * r3.N_DIV + (r3.N_DIV // 2),
        r3.N_DIV ** 3 - 1
    ])

    cell_positions = r3.cell(test_indices)
    re_indexed = r3.index(cell_positions)
    assert (re_indexed == test_indices).all()


def test_lookup():
    translation = torch.randn(100_000, 3)
    translation /= torch.norm(translation, dim=1, keepdim=True)
    translation *= torch.pow(torch.rand(100_000, 1), 1.0 / 3)

    max_trans_distances = r3.distance(r3.cell(r3.index(translation)), translation).max()

    assert max_trans_distances < r3.MAX_DISTANCE_BETWEEN_CELLS


def test_nn():
    test_index = torch.tensor([(r3.N_DIV // 2) * r3.N_DIV ** 2 + (r3.N_DIV // 2)])
    nn_indices = r3.nn(test_index)

    centre_cell = r3.cell(test_index)
    nn_cells = r3.cell(nn_indices)

    distances = r3.distance(centre_cell.unsqueeze(1).repeat(1, 6, 1), nn_cells)

    assert (torch.isclose(distances.abs(), torch.tensor([r3.MAX_DISTANCE_BETWEEN_CELLS])) | (distances == 0)).all()
