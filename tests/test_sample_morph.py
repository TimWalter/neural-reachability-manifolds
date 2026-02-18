import torch

from neural_capability_maps.dataset.morphology import sample_morph, _reject_morph
from neural_capability_maps.dataset.kinematics import forward_kinematics, morph_to_eaik, is_analytically_solvable

torch.set_default_dtype(torch.float64)


def test_reject_degenerate_consecutive_parallel_axes():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, False)

    axes_choice = torch.randint(1, 3, (n_robots,))
    row_indices = torch.arange(n_robots)

    morphs[row_indices, axes_choice, 0] = 0.0
    morphs[row_indices, axes_choice + 1, 0] = 0.0
    morphs[row_indices, axes_choice + 2, 0] = 0.0

    degenerate = _reject_morph(morphs)

    assert degenerate.all(), f"{(~degenerate).sum()} {morphs[~degenerate][0]}"


def test_reject_degenerate_collinear_axes():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, False)

    axes_choice = torch.randint(0, 4, (n_robots,))
    row_indices = torch.arange(n_robots)

    morphs[row_indices, axes_choice, 1:] = torch.tensor([0, 0.1])
    morphs[row_indices, axes_choice + 1, :] = torch.tensor([0, 0, 0.1])

    degenerate = _reject_morph(morphs)
    assert degenerate.all(), f"{(~degenerate).sum()} {morphs[~degenerate][0]}"


def test_analytically_solvable_5dof():
    n_robots = 500
    morphs = sample_morph(n_robots, 5, True)
    for morph in morphs:
        eaik = morph_to_eaik(morph)
        assert eaik.hasKnownDecomposition(), morph


def test_analytically_solvable_6dof():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, True)
    for i, morph in enumerate(morphs):
        eaik = morph_to_eaik(morph)
        assert eaik.hasKnownDecomposition(), f"{i}, {morph}"


def test_is_analytically_solvable_5dof():
    n_robots = 500
    morphs = sample_morph(n_robots, 5, True)
    assert is_analytically_solvable(morphs).all(), morphs[~is_analytically_solvable(morphs)]
    morphs = sample_morph(n_robots, 5, False)
    assert not is_analytically_solvable(morphs).all()


def test_is_analytically_solvable_6dof():
    n_robots = 10
    morphs = sample_morph(n_robots, 6, True)
    assert is_analytically_solvable(morphs).all(), morphs[~is_analytically_solvable(morphs)]
    morphs = sample_morph(n_robots, 6, False)
    assert not is_analytically_solvable(morphs).all()


def test_irrelevant_size():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, False)

    size_factor = torch.rand(n_robots, 1, 1).expand(-1, 7, 2) * 10
    sized_morph = torch.cat((morphs[..., 0:1], size_factor * morphs[..., 1:]), dim=-1)

    joints = 2 * torch.pi * torch.rand(100, *morphs.shape[:-1], 1) - torch.pi
    joints[:, -1, :] = 0

    morph = morphs.unsqueeze(0).expand(100, -1, -1, -1)
    eef_poses = forward_kinematics(morph, joints)[..., -1, :, :]

    sized_morph = sized_morph.unsqueeze(0).expand(100, -1, -1, -1)
    sized_eef_poses = forward_kinematics(sized_morph, joints)[..., -1, :, :]

    orientations = eef_poses[..., :3, :3]
    sized_orientations = sized_eef_poses[..., :3, :3]
    assert (orientations == sized_orientations).all()
    positions = eef_poses[..., :3, 3]
    sized_positions = sized_eef_poses[..., :3, 3]
    size_factor = size_factor[:, 0, 0].unsqueeze(0).unsqueeze(-1).expand(100, -1, 3)
    assert (size_factor * positions - sized_positions).max() < 1e-6
