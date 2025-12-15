import torch
import plotly.graph_objects as go
from torch import Tensor
from jaxtyping import jaxtyped, Float, Bool
from beartype import beartype
import seaborn as sns


def visualise_predictions(poses, pred, gt):
    pred = torch.nn.Sigmoid()(pred) > 0.5
    gt = gt.bool()

    true_positives = pred & gt
    true_negatives = ~pred & ~gt
    false_positives = pred & ~gt
    false_negatives = ~pred & gt

    visualise([poses, poses, poses, poses],
              [true_positives, true_negatives, false_positives, false_negatives],
              names=['True Positives', 'True Negatives', 'False Positives', 'False Negatives'])


def visualise_workspace(poses, labels):
    visualise([poses, poses],
              [labels, ~labels],
              names=['Reachable', 'Unreachable'])


# @jaxtyped(typechecker=beartype)
def visualise(
        poses: list[Float[Tensor, "batch 4 4"]] | Float[Tensor, "batch 4 4"],
        labels: list[Bool[Tensor, "batch"]] | Bool[Tensor, "batch"],
        names: list[str] = None):
    """
    Visualizes SE(3) poses as colored 'L-frames' to show position, orientation, and class
    simultaneously.

    Representation:
    - Origin: End effector position
    - Line: Points backward along -Z axis (direction arm came from)
    - Color: Class Label (e.g., TP/FP or Reachable/Unreachable)

    Args:
        poses: A single or list of batches of SE(3) poses
        labels: A single or list of batches of boolean labels corresponding to the poses.
        names: Optional names for each set of poses for the legend.
    """
    # Normalise inputs to lists
    if isinstance(poses, Tensor):
        poses = [poses]
    if isinstance(labels, Tensor):
        labels = [labels]
    if names is None:
        names = [f"Set {i}" for i in range(len(poses))]

    colors = sns.color_palette("colorblind", n_colors=len(names)).as_hex()

    traces = []

    for i, (pose_batch, label_batch) in enumerate(zip(poses, labels)):
        pose_batch = pose_batch.cpu()
        label_batch = label_batch.cpu()

        subset_poses = pose_batch[label_batch]

        if len(subset_poses) == 0:
            continue

        # 1. Extract Geometry
        origins = subset_poses[:, :3, 3]
        z_axes = subset_poses[:, :3, 2]

        # Calculate start points (pointing back toward robot base)
        z_starts = origins - (z_axes * 0.025)


        # Build line segments: [start, origin, NaN]
        line_segments = torch.stack([z_starts, origins], dim=1)
        # Add NaN separators
        nans = torch.full((line_segments.shape[0], 1, 3), float('nan'))
        with_nans = torch.cat([line_segments, nans], dim=1)

        plot_data = with_nans.reshape(-1, 3).numpy()

        color = colors[i]
        name = names[i]

        legend_group = f"group_{i}"

        traces.append(go.Scatter3d(
            x=plot_data[:, 0],
            y=plot_data[:, 1],
            z=plot_data[:, 2],
            mode='lines',
            line=dict(color=color, width=2),
            name=name,
            legendgroup=legend_group,
            showlegend=True
        ))

        # Optional: Add a small marker at the origin to anchor the eye
        traces.append(go.Scatter3d(
            x=origins[:, 0],
            y=origins[:, 1],
            z=origins[:, 2],
            mode='markers',
            marker=dict(size=2, color=color),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip'
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(
                title='X',
                range=[-1, 1]
            ),
            yaxis=dict(
                title='Y',
                range=[-1, 1]
            ),
            zaxis=dict(
                title='Z',
                range=[-1, 1]
            ),
        ),
        legend=dict(itemsizing='constant', groupclick='togglegroup'),
        paper_bgcolor='white', width=1000, height=1000,
    )
    fig.show()
