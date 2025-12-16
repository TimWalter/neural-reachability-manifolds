import torch
import plotly.graph_objects as go
from torch import Tensor
from jaxtyping import jaxtyped, Float, Bool
from beartype import beartype
import seaborn as sns

from data_sampling.robotics import transformation_matrix, get_capsules


def visualise_predictions(mdh, poses, pred, gt):
    pred = torch.nn.Sigmoid()(pred) > 0.5
    gt = gt.bool()

    true_positives = pred & gt
    true_negatives = ~pred & ~gt
    false_positives = pred & ~gt
    false_negatives = ~pred & gt

    visualise([mdh, mdh, mdh, mdh],
              [poses, poses, poses, poses],
              [true_positives, true_negatives, false_positives, false_negatives],
              names=['True Positives', 'True Negatives', 'False Positives', 'False Negatives'])


def visualise_workspace(mdh, poses, labels):
    visualise([mdh, mdh],
              [poses, poses],
              [labels, ~labels],
              names=['Reachable', 'Unreachable'])


# @jaxtyped(typechecker=beartype)
def visualise(
        mdh: list[Float[Tensor, "dofp1 3"]] | Float[Tensor, "dofp1 3"],
        poses: list[Float[Tensor, "batch 4 4"]] | Float[Tensor, "batch 4 4"],
        labels: list[Bool[Tensor, "batch"]] | Bool[Tensor, "batch"],
        names: list[str] = None):
    """
    Visualizes SE(3) poses as colored 'L-frames' to show position, orientation, and class
    simultaneously.

    Representation:
    - Origin: End effector position
    - Long Line: Points backward along -Z axis (direction EEF came from)
    - Short Line: X-axis (indicates Roll/Orientation)
    - Color: Class Label (e.g., TP/FP or Reachable/Unreachable)

    Args:
        poses: A single or list of batches of SE(3) poses
        labels: A single or list of batches of boolean labels corresponding to the poses.
        names: Optional names for each set of poses for the legend.
    """
    # Normalise inputs to lists
    if isinstance(mdh, Tensor):
        mdh = [mdh]
    if isinstance(poses, Tensor):
        poses = [poses]
    if isinstance(labels, Tensor):
        labels = [labels]
    if names is None:
        names = [f"Set {i}" for i in range(len(poses))]

    colors = sns.color_palette("colorblind", n_colors=len(names)).as_hex()

    traces = []

    for i, (mdh_batch, pose_batch, label_batch) in enumerate(zip(mdh, poses, labels)):
        mdh_batch = mdh_batch.cpu()
        pose_batch = pose_batch.cpu()
        label_batch = label_batch.cpu()

        subset_poses = pose_batch[label_batch]

        if len(subset_poses) == 0:
            continue

        # 1. Extract Geometry
        inv_mat = torch.inverse(
            transformation_matrix(mdh_batch[-1, 0:1], mdh_batch[-1, 1:2], mdh_batch[-1, 2:3], torch.zeros(1)))
        prev_poses = subset_poses @ inv_mat

        origins = subset_poses[:, :3, 3]
        z_axes = prev_poses[:, :3, 2]
        x_axes = prev_poses[:, :3, 0]

        a = mdh_batch[-1, 1]
        d = mdh_batch[-1, 2]

        # Calculate start points (pointing back toward robot base)
        x_ends = origins + (x_axes * 0.025 * (a / (a + d)))
        z_ends = x_ends + (z_axes * 0.025 * (d / (a + d)))

        # Build line segments: [start, origin, NaN]
        l_shapes = torch.stack([x_ends, origins, x_ends, z_ends], dim=1)
        # Add NaN separators
        nans = torch.full((l_shapes.shape[0], 1, 3), float('nan'))
        with_nans = torch.cat([l_shapes, nans], dim=1)

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

        # Show Robot schematically
        legend_group_mdh = f"group_{i}_mdh"
        s_all, e_all = get_capsules(mdh_batch)
        segments = torch.stack([s_all, e_all], dim=1).cpu()
        nans = torch.full((segments.shape[0], 1, 3), float('nan'))
        segments_with_nans = torch.cat([segments, nans], dim=1)
        robot_plot_data = segments_with_nans.reshape(-1, 3)
        traces.append(go.Scatter3d(
            x=robot_plot_data[:, 0],
            y=robot_plot_data[:, 1],
            z=robot_plot_data[:, 2],
            mode='lines',
            line=dict(color=color, width=8),
            name=f"Robot ({names[i]})",
            legendgroup=legend_group_mdh,
            showlegend=True,
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
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)  # Default is ~1.25; smaller = closer
            )
        ),
        legend=dict(orientation="h",  # Horizontal legend
                    yanchor="bottom",
                    y=1.02,  # Places legend above the plot
                    xanchor="right",
                    x=1,
                    itemsizing='constant',
                    groupclick='togglegroup',
                    bgcolor='rgba(255,255,255,0.5)'  # Semi-transparent background
                    ),
        paper_bgcolor='white', width=1000, height=1000,
    )
    fig.show()
