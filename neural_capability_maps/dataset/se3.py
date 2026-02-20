import math
import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float, Int64

import neural_capability_maps.dataset.r3 as r3
import neural_capability_maps.dataset.so3 as so3

LEVEL, MIN_DISTANCE_BETWEEN_CELLS, MAX_DISTANCE_BETWEEN_CELLS, N_CELLS = [None] * 4


def set_level(level: int = 3):
    global LEVEL, MIN_DISTANCE_BETWEEN_CELLS, MAX_DISTANCE_BETWEEN_CELLS, N_CELLS
    LEVEL = level
    r3.set_level(level)
    so3.set_level(level)
    MIN_DISTANCE_BETWEEN_CELLS = math.sqrt(r3.DISTANCE_BETWEEN_CELLS ** 2 / 8 +
                                           so3.MIN_DISTANCE_BETWEEN_CELLS ** 2 / (2 * torch.pi ** 2))
    MAX_DISTANCE_BETWEEN_CELLS = math.sqrt(r3.DISTANCE_BETWEEN_CELLS ** 2 / 8 +
                                           so3.MAX_DISTANCE_BETWEEN_CELLS ** 2 / (2 * torch.pi ** 2))
    N_CELLS = r3.N_CELLS * so3.N_CELLS


set_level()


# @jaxtyped(typechecker=beartype)
def distance(x1: Float[Tensor, "*batch 4 4"], x2: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 1"]:
    r"""
    Pose distance arising from the unique left-invariant riemannian metric for SE(3) that produces physically meaningful
    accelerations plus a weighting between translation and rotation.

    Args:
        x1: First homogeneous transformation.
        x2: Second homogeneous transformation.

    Returns:
        SE(3) distance between x1 and x2.

    Notes:
        Since the maximum rotational distance is \pi and the maximum translational distance in our setting is 2,
        we weigh the distances "equal" importance and also such that the maximum distance between two cells is 1.
    """
    t1 = x1[..., :3, 3]
    r1 = x1[..., :3, :3]
    t2 = x2[..., :3, 3]
    r2 = x2[..., :3, :3]
    return torch.sqrt(r3.distance(t1, t2) ** 2 / 8 + so3.distance(r1, r2) ** 2 / (2 * torch.pi ** 2))


# @jaxtyped(typechecker=beartype)
def split_index(index: Int64[Tensor, "*batch"]) -> tuple[Int64[Tensor, "*batch"], Int64[Tensor, "*batch"]]:
    """
    Split SE(3) cell index into R3 and SO(3) indices.

    Args:
        index: SE(3) cell index.

    Returns:
        R3 and SO(3) index
    """
    return index % r3.N_CELLS, index // r3.N_CELLS


# @jaxtyped(typechecker=beartype)
def combine_index(r3_index: Int64[Tensor, "*batch"], so3_index: Int64[Tensor, "*batch"]) -> Int64[Tensor, "*batch"]:
    """
    Combine R3 and SO(3) index into SE(3) cell index.

    Args:
        r3_index: R3 index.
        so3_index: SO(3) index.

    Returns:
        SE(3) cell index.
    """
    return r3_index + so3_index * r3.N_CELLS


# @jaxtyped(typechecker=beartype)
def index(pose: Float[Tensor, "*batch 4 4"]) -> Int64[Tensor, "*batch"]:
    """
    Get cell index for the given poses.

    Args:
        pose: Pose.

    Returns:
        Cell index.
    """
    return combine_index(r3.index(pose[:, :3, 3]), so3.index(pose[:, :3, :3]))


# @jaxtyped(typechecker=beartype)
def cell(index: Int64[Tensor, "*batch"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Get cell pose for the given index.

    Args:
        index: Cell index.

    Returns:
        Cell pose.
    """
    r3_index, so3_index = split_index(index)
    pose = torch.eye(4, device=index.device).repeat(*index.shape, 1, 1)
    pose[..., :3, 3] = r3.cell(r3_index)
    pose[..., :3, :3] = so3.cell(so3_index)
    return pose


# @jaxtyped(typechecker=beartype)
def cell_noisy(index: Int64[Tensor, "*batch"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Get cell pose for the given index, with noise, such that not the centre but any pose in the cell is queried.

    Args:
        index: Cell index

    Returns:
        Cell pose
    """
    r3_index, so3_index = split_index(index)
    pose = torch.eye(4, device=index.device).repeat(*index.shape, 1, 1)
    pose[..., :3, 3] = r3.cell_noisy(r3_index)
    pose[..., :3, :3] = so3.cell_noisy(so3_index)
    return pose


# @jaxtyped(typechecker=beartype)
def nn(index: Int64[Tensor, "*batch"]) -> Int64[Tensor, "*batch 12"]:
    """
    Get nearest neighbour cell indices for the given index.

    Args:
        index: Cell index

    Returns:
        Nearest neighbour cell indices

    Notes:
        For boundary cells, we return the index of the cell itself for the out-of-bounds neighbours.
    """
    r3_index, so3_index = split_index(index)

    nn_r3, nn_so3 = split_index(index.unsqueeze(-1).repeat(*([1] * index.ndim), 12))
    nn_r3[..., :6] = r3.nn(r3_index)
    nn_so3[..., 6:] = so3.nn(so3_index)

    nn = combine_index(nn_r3, nn_so3)
    return nn


# @jaxtyped(typechecker=beartype)
def random(num_samples: int) -> Float[Tensor, "num_samples 4 4"]:
    """
    Sample random poses uniformly from SE(3).

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Random poses.
    """
    pose = torch.eye(4).repeat(num_samples, 1, 1)
    pose[:, :3, :3] = so3.random(num_samples)
    pose[:, :3, 3] = r3.random(num_samples)
    return pose


# @jaxtyped(typechecker=beartype)
def random_ball(num_samples: int,
                centre: Float[Tensor, "3"],
                radius: float | Float[Tensor, "1"]) -> Float[Tensor, "num_samples 4 4"]:
    """
    Sample random poses uniformly from a bounding ball.

    Args:
        num_samples: Number of samples to generate.
        centre: Ball centre.
        radius: Ball radius.

    Returns:
        Random poses.
    """
    pose = torch.eye(4).repeat(num_samples, 1, 1)
    pose[:, :3, :3] = so3.random(num_samples)
    pose[:, :3, 3] = r3.random_ball(num_samples, centre, radius)
    return pose.to(centre.device)


# @jaxtyped(typechecker=beartype)
def to_vector(pose: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 9"]:
    """
    Convert 4x4 pose represented by a homogeneous transformation matrix to a 9D vector representation.

    Args:
        pose: Homogeneous transformation matrix

    Returns:
        9D vector representation
    """
    return torch.cat([pose[..., :3, 3], so3.to_vector(pose[..., :3, :3])], dim=-1)


# @jaxtyped(typechecker=beartype)
def from_vector(vec: Float[Tensor, "*batch 9"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Convert 9D vector representation to 4x4 homogeneous transformation matrix

    Args:
        vec: 9D vector representation

    Returns:
        Homogeneous transformation matrix
    """
    translation = vec[..., :3]
    rotation_cont = vec[..., 3:]
    batch_shape = vec.shape[:-1]
    homogeneous = torch.eye(4, device=vec.device).expand(*batch_shape, 4, 4).clone()
    homogeneous[..., :3, 3] = translation
    homogeneous[..., :3, :3] = so3.from_vector(rotation_cont)
    return homogeneous


# @jaxtyped(typechecker=beartype)
def exp(pose: Float[Tensor, "*batch 4 4"], tangent: Float[Tensor, "*batch 6"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Differential geometry version of addition.

    Args:
        pose: Pose.
        tangent: Tangent vector.

    Returns:
        Moves from pose along the tangent vector.

    Notes:
        In Euclidean space, 𝑎𝑑𝑑𝑖𝑡𝑖𝑜𝑛 is a tool which takes two points 𝑝1,𝑝2, “adds” them, and generates a third, larger point
        𝑝3. Addition gives us a way to “move forward” in Euclidean space. On manifolds, the 𝑒𝑥𝑝𝑜𝑛𝑒𝑛𝑡𝑖𝑎𝑙 provides a tool,
        which “takes the exponential of the tangent vector at point 𝑝” to generate a third point on the manifold.
        The exponential does this by
        1) identifying the unique geodesic 𝛾 that goes through 𝑝 and 𝑣𝑝,
        2) identifying the “length” 𝑙 of the tangent vector 𝑣𝑝, and
        3) calculating another point 𝑝′ along 𝛾⁡(𝑡) that is a “distance” 𝑙 from the initial point 𝑝.
        Note again that the notion of “length” and “distance” is different on a manifold than it is in Euclidean space
        and that quantifying length is not something that we will be able to do without specifying a metric.
        [Source https://geomstats.github.io/notebooks/02_foundations__connection_riemannian_metric.html]
    """
    pose = pose.clone()
    pose[..., :3, 3] = r3.exp(pose[..., :3, 3], tangent[..., :3])
    pose[..., :3, :3] = so3.exp(pose[..., :3, :3], tangent[..., 3:])
    return pose


# @jaxtyped(typechecker=beartype)
def log(pose1: Float[Tensor, "*batch 4 4"], pose2: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 4 4"]:
    """
    Differential geometry version of addition.

    Args:
        pose1: First pose.
        pose2: Second pose.

    Returns:
        Logarithmic map of the second pose relative to the first.

    Notes:
        In Euclidean space, 𝑠𝑢𝑏𝑡𝑟𝑎𝑐𝑡𝑖𝑜𝑛 is an operation which allows us to take the third point 𝑝3 and one of the
        initial points 𝑝1 and extract the other initial point 𝑝2. Similarly, the 𝑙𝑜𝑔𝑎𝑟𝑖𝑡ℎ𝑚 allows us to take the
        final point 𝑝′ and the initial point 𝑝 to extract the tangent vector 𝑣𝑝 at the initial point.
        The logarithm is able to do this by
        1) identifying the unique geodesic 𝛾 that connects the two points
        2) calculating the “length” of that geodesic
        3) generating the unique tangent vector at 𝑝, with a “length” equal to that of the geodesic.
        Again, remember that “length” is not something that we can quantify without specifying a metric.
        A key point here is that if you know a point and a tangent vector at that point, you can calculate a unique
        geodesic that goes through that point. Similarly, if you know the point and geodesic, you should be able to
        extract the unique tangent vector that produced that geodesic.
        [Source https://geomstats.github.io/notebooks/02_foundations__connection_riemannian_metric.html]
    """
    return torch.cat([
        r3.log(pose1[..., :3, 3], pose2[..., :3, 3]),
        so3.log(pose1[..., :3, :3], pose2[..., :3, :3])
    ], dim=-1)
