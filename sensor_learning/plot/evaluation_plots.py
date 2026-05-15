#!usr/bin/env python3
import plotly.graph_objects as go
import numpy as np



def plot_kd_tree_coverage(curr_pose, poses, kd_idxs, fig: go.Figure | None = None):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=poses[kd_idxs][:, 0],
            y=poses[kd_idxs][:, 1],
            z=poses[kd_idxs][:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="closest poses",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=curr_pose[0:1],
            y=curr_pose[1:2],
            z=curr_pose[2:3],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="current pose",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=poses[:, 0],
            y=poses[:, 1],
            z=poses[:, 2],
            mode="markers",
            marker=dict(size=2, color="lightgray"),
            name="all poses",   
        )
    )
    return fig  


def update_tree_coverage_plot(
    curr_pose_idx: int, 
    kd_idxs: np.ndarray, 
    poses: np.ndarray, 
    detected_faces: np.ndarray, 
    vertices: np.ndarray, 
    colors: np.ndarray, 
    observation_counts: np.ndarray
) -> go.Figure:
    curr_pose = poses[curr_pose_idx]
    fig = plot_kd_tree_coverage(curr_pose, poses, kd_idxs)

    

    vertices = vertices[detected_faces]
    colors = colors[detected_faces]
    observation_counts = observation_counts[detected_faces]
    n_triangles = vertices.shape[0]
    flat_vertices = vertices.reshape(-1, 3)
    indices = np.arange(n_triangles * 3)

    # set opacity based on observation counts (1 - 5 counts normalized to [0.1, 1.0])
    alphas = np.clip(observation_counts / 5.0, 0.1, 1.0)
    alphas = alphas.tolist()
    
    fig.add_trace(
        go.Mesh3d(
            x=flat_vertices[:, 0],
            y=flat_vertices[:, 1],
            z=flat_vertices[:, 2],
    
            i=indices[0::3],
            j=indices[1::3],
            k=indices[2::3],
    
            facecolor=[f'rgba({color[0]},{color[1]},{color[2]},{alpha})' for color, alpha in zip(colors, alphas)],
        )
    )
    fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=2, y=2, z=2), uirevision=True))
    return fig


def create_error_figure(message: str = "ERROR") -> go.Figure:
    """
    Create a Plotly 3D scatter figure displaying an error message.
    """

    fig = go.Figure()

    # Invisible point just to initialize the 3D scene
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=1, opacity=0),
            showlegend=False,
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=32, color="red"),
            )
        ],
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig

if __name__ == "__main__":
    continuous_tree_plotter()