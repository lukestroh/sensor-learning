#!/usr/bin/env python3
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import threading
from sensor_learning.plot.evaluation_plots import update_tree_coverage_plot, create_error_figure
import time
import plotly.graph_objects as go


app = Dash(__name__)

class TreeCoverageApp:
    def __init__(self):
        self._lock = threading.Lock()
        self._curr_pose_idx = None
        self._poses = None
        self._vertices = None
        self._colors = None
        self._observation_counts = None
        self._kd_idxs = None

        # dash app
        self.app = Dash(__name__)
        
        self.app.layout = html.Div([
                dcc.Graph(
                    id="live-graph",
                    style={
                        "width": "100vw",
                        "height": "100vh",
                    },
                ),
                dcc.Interval(
                    id="interval",
                    interval=1000, # ms
                    n_intervals=0,
                ),
            ],
            style={
                    "margin": "0",
                    "padding": "0",
                    "width": "100vw",
                    "height": "100vh",
                },
        )

        self.app.callback(
            Output("live-graph", "figure"),
            Input("interval", "n_intervals"),
        )(self._dash_update)
        return

    def _dash_update(self, n_intervals) -> go.Figure:
        with self._lock:
            if all(
                x is not None
                for x in (
                    self._curr_pose_idx,
                    self._kd_idxs,
                    self._poses,
                    self._detected_faces,
                    self._vertices,
                    self._colors,
                    self._observation_counts,
                )
            ):
                fig = update_tree_coverage_plot(
                    curr_pose_idx=self._curr_pose_idx,
                    kd_idxs=self._kd_idxs,
                    poses=self._poses,
                    detected_faces=self._detected_faces,
                    vertices=self._vertices,
                    colors=self._colors,
                    observation_counts=self._observation_counts,
                )
                # fig.show()
                return fig
               
        return create_error_figure() # Return an empty figure if data is not ready

    def update(self,
        curr_pose_idx,
        kd_idxs,
        poses,
        detected_faces: np.ndarray,
        vertices,
        colors,
        observation_counts
    ):
        with self._lock:
            self._curr_pose_idx = curr_pose_idx
            self._kd_idxs = kd_idxs
            self._poses = poses
            self._detected_faces = detected_faces
            self._vertices = vertices
            self._colors = colors
            self._observation_counts = observation_counts
        return
       
    def run(self):
        self.app.run(debug=True, use_reloader=False)  # Set use_reloader to False to prevent multiple instances
        return 