import torch

import data_sampling.representations as repr
import data_sampling.so3 as so3
import data_sampling.se3 as se3


def test_cont_and_rotation_matrix():
    rot_mat = so3.random(100)

    cont = repr.rotation_matrix_to_continuous(rot_mat)
    rot_mat_reconstructed = repr.continuous_to_rotation_matrix(cont)

    assert torch.allclose(rot_mat, rot_mat_reconstructed)

def test_homogeneous_and_vector():
    homogeneous = se3.random(100)

    vec = repr.homogeneous_to_vector(homogeneous)
    homogeneous_reconstructed = repr.vector_to_homogeneous(vec)

    assert torch.allclose(homogeneous, homogeneous_reconstructed)