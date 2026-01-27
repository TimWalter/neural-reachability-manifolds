import torch
from torch import Tensor
from jaxtyping import Float, Int64, jaxtyped
from beartype import beartype


# @jaxtyped(typechecker=beartype)
def distance(x1: Float[Tensor, "*batch 3"], x2: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 1"]:
    """
    Euclidean distance between vectors.

    Args:
        x1: First vector.
        x2: Second vector.

    Returns:
        Euclidean distance between vector x1 and x2.
    """
    return torch.norm(x1 - x2, dim=-1, keepdim=True)


N_DIV = 36
MAX_DISTANCE_BETWEEN_CELLS = 2 / N_DIV
N_CELLS = N_DIV ** 3


# @jaxtyped(typechecker=beartype)
def _split_index(index: Int64[Tensor, "*batch"]) -> tuple[
    Int64[Tensor, "*batch"], Int64[Tensor, "*batch"], Int64[Tensor, "*batch"]]:
    x = index % N_DIV
    y = (index // N_DIV) % N_DIV
    z = index // (N_DIV * N_DIV)
    return x, y, z


# @jaxtyped(typechecker=beartype)
def _combine_index(x: Int64[Tensor, "*batch"], y: Int64[Tensor, "*batch"], z: Int64[Tensor, "*batch"]) \
        -> Int64[Tensor, "*batch"]:
    return x + y * N_DIV + z * N_DIV * N_DIV


# @jaxtyped(typechecker=beartype)
def index(position: Float[Tensor, "*batch 3"]) -> Int64[Tensor, "*batch"]:
    """
    Get cell index for the given position.

    Args:
        position: Position

    Returns:
        Cell index
    """
    indices = torch.floor((position + 1.0) / 2.0 * N_DIV).to(torch.int64)
    indices = torch.clamp(indices, 0, N_DIV - 1)
    return _combine_index(indices[..., 0], indices[..., 1], indices[..., 2])


# @jaxtyped(typechecker=beartype)
def cell(index: Int64[Tensor, "*batch"]) -> Float[Tensor, "*batch 3"]:
    """
    Get cell position for the given index.

    Args:
        index: Cell index

    Returns:
        Cell position
    """
    x, y, z = _split_index(index)
    indices = torch.stack([x, y, z], dim=-1)
    cell = ((indices + 0.5) / N_DIV) * 2.0 - 1.0
    return cell


# @jaxtyped(typechecker=beartype)
def nn(index: Int64[Tensor, "*batch"]) -> Int64[Tensor, "*batch 6"]:
    """
    Get nearest neighbour cell indices for the given index.

    Args:
        index: Cell index

    Returns:
        Nearest neighbour cell indices

    Notes:
        For boundary cells, we return the index of the cell itself for the out-of-bounds neighbours.
    """
    nn = index.unsqueeze(-1).repeat(*([1] * index.ndim), 6)

    x, y, z = _split_index(index)

    nn[x != N_DIV - 1, 0] += 1  # +x
    nn[x != 0, 1] -= 1  # -x
    nn[y != N_DIV - 1, 2] += N_DIV  # +y
    nn[y != 0, 3] -= N_DIV  # -y
    nn[z != N_DIV - 1, 4] += N_DIV * N_DIV  # +z
    nn[z != 0, 5] -= N_DIV * N_DIV  # -z

    return nn


# @jaxtyped(typechecker=beartype)
def random(num_samples: int) -> Float[Tensor, "num_samples 3"]:
    """
    Sample random positions uniformly from R3.

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Random positions.
    """
    translation = torch.randn(num_samples, 3)
    translation /= torch.norm(translation, dim=1, keepdim=True)
    translation *= torch.pow(torch.rand(num_samples, 1), 1.0 / 3)

    return translation


# @jaxtyped(typechecker=beartype)
def random_ball(num_samples: int,
                centre: Float[Tensor, "3"],
                radius: float) -> Float[Tensor, "num_samples 3"]:
    """
    Sample random positions uniformly from a bounding ball.

    Args:
        num_samples: Number of samples to generate.
        centre: Ball centre.
        radius: Ball radius.

    Returns:
        Random positions.
    """
    direction = torch.randn(num_samples, 3, device=centre.device)
    direction /= torch.norm(direction, dim=1, keepdim=True)
    translation = centre + radius * torch.rand(num_samples, device=centre.device).unsqueeze(1) ** (1 / 3) * direction
    return translation


# @jaxtyped(typechecker=beartype)
def exp(position: Float[Tensor, "*batch 3"], tangent: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
    """
    Differential geometry version of addition.

    Args:
        position: Position.
        tangent: Tangent vector.

    Returns:
        Moves from position along the tangent vector.

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
    return position + tangent


# @jaxtyped(typechecker=beartype)
def log(position1: Float[Tensor, "*batch 3"], position2: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
    """
    Differential geometry version of subtraction.

    Args:
        position1: First position.
        position2: Second position.

    Returns:
        The tangent vector at position1 pointing to position2.

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
    return position2 - position1
