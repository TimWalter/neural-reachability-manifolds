import torch
from data_sampling.se3_cells import R3_CELLS, R3_MAX_DISTANCE_BETWEEN_CELLS, translational_distance, \
    SO3_CELLS, SO3_MAX_DISTANCE_BETWEEN_CELLS, rotational_distance, r3_indices, so3_indices
from data_sampling.orientation_representations import quaternion_to_rotation_matrix

def test_verify_r3_max_distance_between_cells():
    distances = translational_distance(
        R3_CELLS.unsqueeze(0).expand(R3_CELLS.shape[0], R3_CELLS.shape[0], 3),
        R3_CELLS.unsqueeze(1).expand(R3_CELLS.shape[0], R3_CELLS.shape[0], 3)).squeeze(-1)
    distances = distances.sort(dim=1).values  # the smallest distance is with themselves
    assert torch.allclose(distances[:, 1].max(), torch.tensor([R3_MAX_DISTANCE_BETWEEN_CELLS]))


def test_verify_s03_max_distance_between_cells():
    distances = rotational_distance(
        SO3_CELLS.unsqueeze(0).expand(SO3_CELLS.shape[0], SO3_CELLS.shape[0], 3, 3),
        SO3_CELLS.unsqueeze(1).expand(SO3_CELLS.shape[0], SO3_CELLS.shape[0], 3, 3)).squeeze(-1)
    distances = distances.sort(dim=1).values  # the smallest distance is with themselves
    assert torch.allclose(distances[:, 1].max(), torch.tensor([SO3_MAX_DISTANCE_BETWEEN_CELLS]))

def test_verify_r3_lookup():
    translation = torch.randn(100_000, 3)
    translation /= torch.norm(translation, dim=1, keepdim=True)
    translation *= torch.pow(torch.rand(100_000, 1), 1.0 / 3)

    max_trans_distances = translational_distance(R3_CELLS[r3_indices(translation).cpu()], translation).max()

    assert max_trans_distances < R3_MAX_DISTANCE_BETWEEN_CELLS

def test_verify_so3_lookup():
    quaternions = torch.randn(100_000, 4)
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    rotation = quaternion_to_rotation_matrix(quaternions)

    max_rot_distances = rotational_distance(SO3_CELLS[so3_indices(rotation).cpu()], rotation).max()

    assert max_rot_distances < SO3_MAX_DISTANCE_BETWEEN_CELLS
