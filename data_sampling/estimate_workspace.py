import math
import torch
import faiss
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int

from data_sampling.autotune_batch_size import get_batch_size
from data_sampling.robotics import forward_kinematics, collision_check, analytical_inverse_kinematics

import data_sampling.se3 as se3
import data_sampling.so3 as so3

torch.set_float32_matmul_precision('high')


# @jaxtyped(typechecker=beartype)
def sample_joints(batch_size: int, dofp1: int) -> Float[Tensor, "{batch_size} {dof+1}"]:
    """
    sample random joint configurations for the robot.

    args:
        batch_size: Number of joint configurations to sample
        dofp1: Degrees of freedom of the robot + 1

    returns:
        Newly sampled joint configurations
    """
    joints = torch.cat([2 * torch.pi * torch.rand(batch_size, dofp1 - 1, device="cuda") - torch.pi,
                        torch.zeros(batch_size, 1, device="cuda")], dim=1)

    return joints


# @jaxtyped(typechecker=beartype)
def workload(batch_size: int, morph: Float[Tensor, "dof 3"]) -> Int[Tensor, "n_cells"]:
    """
    Computes one batch of the capability map estimation.
    Args:
        batch_size: Number of samples to process in this batch
        morph: MDH parameters encoding the robot geometry

    Returns:
        Cell indices of reachable cells

    Notes:
        Cells with self-collisions are not necessarily unreachable, as the max operator is applied for a pose on all
        joints configurations
    """
    new_joints = sample_joints(batch_size, morph.shape[0])
    new_poses = forward_kinematics(morph.unsqueeze(0).expand(batch_size, -1, -1), new_joints.unsqueeze(-1))
    self_collision = collision_check(morph.unsqueeze(0).expand(new_poses.shape[0], -1, -1), new_poses)

    new_cell_indices = se3.index(new_poses[:, -1, :, :][~self_collision])

    return new_cell_indices


compiled_workload = torch.compile(workload)


# @jaxtyped(typechecker=beartype)
def estimate_workspace(morph: Float[Tensor, "dofp1 3"]) -> tuple[
    Int[Tensor, "n_cells"],
    Int[Tensor, "n_cells"]
]:
    """
    Sample the robot's workspace and return the estimated capability map.

    Args:
        morph: MDH parameters encoding the robot geometry

    Returns:
        Labels and indices encoding the discretized capability map
    """
    morph = morph.to("cuda")

    pos_indices = torch.empty(0, dtype=torch.int64, device="cuda")

    args = (morph,)
    batch_size = get_batch_size(morph.device, workload, args, safety=0.5)

    filled_cells = 0
    newly_filled_cells = 0

    while (newly_filled_cells == 0 and filled_cells == 0) or newly_filled_cells / filled_cells > 1e-3:
        new_pos_indices = workload(batch_size, *args)

        pos_indices = torch.cat([pos_indices, new_pos_indices]).unique()

        newly_filled_cells = len(pos_indices) - filled_cells
        filled_cells += newly_filled_cells

    pos_indices = pos_indices.cpu()

    # neg_indices = torch.empty(0, dtype=torch.int64)
    # while neg_indices.shape[0] != pos_indices.shape[0]:
    #     neg_indices = torch.cat([neg_indices, torch.randint(0, se3.N_CELLS, (2*(pos_indices.shape[0] - neg_indices.shape[0]),))])
    #     neg_indices = neg_indices.unique()
    #     neg_indices = neg_indices[~torch.isin(neg_indices, pos_indices)]
    #     neg_indices = neg_indices[:pos_indices.shape[0]]

    # Boundary as negative samples
    neg_indices = se3.nn(pos_indices).flatten().unique()
    neg_indices = neg_indices[~torch.isin(neg_indices, pos_indices)]

    # Morphological gradient on positive indices
    # neighbours = se3.nn(pos_indices)
    # pos_indices = pos_indices[~torch.isin(neighbours, pos_indices.unsqueeze(0)).all(dim=1)]

    cell_indices = torch.cat([pos_indices, neg_indices], dim=0)
    labels = torch.cat([torch.ones_like(pos_indices), torch.zeros_like(neg_indices)], dim=0)

    return labels, cell_indices


# @jaxtyped(typechecker=beartype)
def create_index(poses: Float[Tensor, "N 4 4"]) -> faiss.IndexIVFFlat:
    """
    Create a faiss IndexIVFFlat from the given poses for faster approximate nearest neighbour search.

    Args:
        poses: SE3 poses
    Returns:
        faiss index

    Notes:
        Since only the Euclidean distance is implemented, we do ANN search only in regard to the position.
    """
    poses_embed = poses[:, :3, 3].contiguous()

    quantizer = faiss.IndexFlatL2(poses_embed.shape[1])
    index = faiss.IndexIVFFlat(quantizer, poses_embed.shape[1], int(math.sqrt(poses_embed.shape[0])), faiss.METRIC_L2)

    train_idx = torch.randperm(poses_embed.shape[0])[:min(500_000, poses_embed.shape[0])]
    train_data = poses_embed[train_idx]

    index.train(train_data)
    index.add(poses_embed)

    return index


# @jaxtyped(typechecker=beartype)
def predict(index: faiss.Index,
            train_labels: Float[Tensor, "N"],
            train_poses: Float[Tensor, "N 4 4"],
            query_poses: Float[Tensor, "batch 4 4"],
            k: int = so3.N_CELLS) -> Float[Tensor, "batch"]:
    """
    Query FAISS index and perform interpolation between the min(13, k) nearest neighbours. We oversample
    the nearest neighbours since the l2 distance on the positions is only a proxy for the actual SE3 distance.

    Args:
        index: faiss index
        train_labels: labels for all indexed poses
        train_poses: all indexed poses
        query_poses: queried poses
        k: number of nearest neighbours to query
    Returns:
        predicted labels for queried poses
    """
    nn = torch.from_numpy(index.search(query_poses[:, :3, 3].numpy(), k)[1]).to(torch.long)

    nn_labels = torch.cat([train_labels[nn[:, idx]].unsqueeze(1) for idx in range(nn.shape[1])], dim=1)
    nn_poses = torch.cat([train_poses[nn[:, idx]].unsqueeze(1) for idx in range(nn.shape[1])], dim=1)
    nn_distances = torch.cat([se3.distance(nn_poses[:, idx, :, :], query_poses) for idx in range(nn.shape[1])], dim=1)

    prune_idx = nn_distances.argsort(dim=1)[:, :13]
    nn_labels = torch.gather(nn_labels, 1, prune_idx)
    nn_distances = torch.gather(nn_distances, 1, prune_idx)

    weights = 1 / (nn_distances + 1e-6) ** 6
    weights /= (weights.sum(dim=1, keepdim=True) + 1e-12)

    query_labels = (nn_labels * weights).sum(dim=1)
    query_labels[query_labels < 0.5] = 0.0
    query_labels[query_labels >= 0.5] = 1.0

    return query_labels.long()


def estimate_workspace_analytically(morph: Float[Tensor, "dofp1 3"], num_samples: int) -> tuple[
    Int[Tensor, "num_samples 1"],
    Float[Tensor, "num_samples 4 4"]
]:
    """
    Estimate the workspace of a robot with analytically solvable inverse kinematics.

    Args:
        morph: MDH parameters encoding the robot geometry
        num_samples: Number of samples to generate
    Returns:
        Labels and poses encoding the discretised capability map (Balanced classes)
    """

    poses = torch.eye(4).repeat(num_samples, 1, 1)
    labels = torch.zeros(num_samples, dtype=torch.float32)

    idx = 0
    # Generate unreachable samples by uniformly sampling the workspace and verifying via IK
    while idx != num_samples // 2:
        poses[idx:num_samples // 2] = se3.random(num_samples // 2 - idx)
        joints, manipulability = analytical_inverse_kinematics(morph, poses[idx:num_samples // 2])
        mask = manipulability == -1
        added_c = (mask).sum()
        poses[idx:idx + added_c] = poses[idx:num_samples // 2][mask]
        idx += added_c
    # Generate reachable samples via FK
    morph = morph.to("cuda")
    while idx != num_samples:
        num_c = 2 * (num_samples - idx)
        joints = sample_joints(num_c, morph.shape[0])
        reached_poses = forward_kinematics(morph.unsqueeze(0).expand(num_c, -1, -1), joints.unsqueeze(-1))
        self_collision = collision_check(morph.unsqueeze(0).expand(reached_poses.shape[0], -1, -1), reached_poses)
        reached_poses = reached_poses[~self_collision]

        added_c = min((~self_collision).sum().cpu(), num_samples - idx)
        reached_poses = reached_poses[:added_c]

        poses[idx:idx + added_c] = reached_poses[:, -1, :, :].cpu()
        labels[idx:idx + added_c] = torch.ones(reached_poses.shape[0])
        idx += added_c
    return labels.unsqueeze(1), poses


if __name__ == "__main__":
    from data_sampling.sample_morph import sample_morph

    torch.manual_seed(0)
    morphs = [sample_morph(1, i, True)[0] for i in range(1, 7)]

    labels, cell_indices = estimate_workspace_analytically(morphs[4], 100_000)
