#!/usr/bin/env python3

import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation


def plot_quaternions(quaternions):
    """Plot quaternions as unit vectors

    :param quaternions: _description_
    :type quaternions: _type_
    """

    fig = go.Figure()

    world_z = np.array([0, 0, 1])
    rotations = Rotation.from_quat(quaternions).as_matrix()
    unit_vectors = rotations @ world_z

    origins = np.zeros(unit_vectors.shape)

    for i in range(unit_vectors.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=[origins[i, 0], unit_vectors[i, 0]],
                y=[origins[i, 1], unit_vectors[i, 1]],
                z=[origins[i, 2], unit_vectors[i, 2]],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        title="Quaternion Unit Vectors",
    )

    return fig


def plot_vectors(vectors):
    """Plot 3D vectors

    :param vectors: _description_
    :type vectors: _type_
    """

    fig = go.Figure()

    vecs = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    for vec in vecs:
        fig.add_trace(
            go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=2),
            )
        )
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"), title="3D Vectors"
    )
    return fig


def plot_positions(positions):
    """Plot 3D positions

    :param positions: _description_
    :type positions: _type_
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(size=4),
        )
    )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"), title="3D Positions"
    )

    return fig
