#!/usr/bin/env python3
import pprint as pp
from dataclasses import dataclass

import h5py as h5
import numpy as np
import plotly.graph_objects as go
import scipy.spatial as sps
from plotly import data
from pybullet_tree_sim.sensors.sensor import st
from pybullet_tree_sim.tree_metadata import Face
from scipy.constants import c
import os
from sensor_learning.plot import evaluation_plots as eval_plots
from sensor_learning.utils import pose_generator


@dataclass
class ObservationConfig:
    min_distance: float  # meters, get as input from z_near
    max_distance: float  # meters, get as input from z_far
    max_incidence_angle: float = 45.0  # degrees
    kd_radius: float = 0.1  # radius for finding neighboring poses in kd-tree search


@dataclass
class GainConfig:
    saturation_lambda: float = 1.0  # controls how fast gain saturates
    n_midpoint: int = 5  # observations beyon n_max add little value
    steepness: float = 2.0  # controls steepness of gain curve around n_midpoint


def key_tree(hfile):
    if hasattr(hfile, "keys"):
        # Group node — recurse into children
        return {key: key_tree(hfile[key]) for key in hfile.keys()}
    elif hasattr(hfile, "read_direct"):
        # Dataset node — read into numpy array
        value = hfile[:]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif value.dtype.kind == "S":  # byte-string array
            value = value.astype(str)
        attrs = {attr: hfile.attrs[attr] for attr in hfile.attrs}
        return {"data": value, "attrs": attrs} if attrs else value
    else:
        return hfile


class CoverageEvaluator:
    def __init__(self, data_dict: dict, observation_config: ObservationConfig, gain_config: GainConfig) -> None:
        # print(data_dict['eef_poses'].keys())
        self.data = data_dict
        self.poses = self.data["eef_poses"]["actual_poses"]  # type: ignore
        self.kd_tree = sps.KDTree(self.poses[:, :3])  # type: ignore
        # print(self.kd_tree)
        self.observation_config = observation_config
        self.gain_config = gain_config
        self.faces = self.data["tree"]["faces"]
        self.face_colors = self.faces["color"]
        self.observation_counts = np.zeros(shape=self.face_colors.shape[0], dtype=np.int32)
        self.pose_gains = np.zeros(shape=self.data["eef_poses"]["actual_poses"].shape[0], dtype=np.float32)
        self.total_gain = 0.0
        return

    def get_detected_faces_from_sensor(self, pose_idx: int, pose_data: dict, sensor_data: dict) -> np.ndarray | None:
        # extrinsics = sensor_data['extrinsics'][pose_idx]
        depth_rgb = sensor_data["depth"]["depth_rgb"][pose_idx]
        depth = sensor_data['depth']['depth'][pose_idx]
        is_black = np.all(depth_rgb == 0, axis=2)
        hs, ws = np.where(~is_black)
        pixels = depth_rgb[hs, ws]
        # print(pixels)
        detected_colors_packed = (
            (pixels[:, 0].astype(np.uint32) << 16)
            | (pixels[:, 1].astype(np.uint32) << 8)
            | pixels[:, 2].astype(np.uint32)
        )

        face_colors_packed = (
            (self.face_colors[:, 0].astype(np.uint32) << 16)
            | (self.face_colors[:, 1].astype(np.uint32) << 8)
            | self.face_colors[:, 2].astype(np.uint32)
        )
        matches = face_colors_packed[:, None] == detected_colors_packed[None, :]
        detected_faces = np.where(matches)

        # discount faces that are more than max incidence angle (should be 45 degrees)
        normals = self.faces["normal"][detected_faces[0]]
        actual_pose = pose_data[pose_idx]
        actual_pose_ori = sps.transform.Rotation.from_quat(actual_pose[3:]).as_matrix()
        actual_pose_vec = actual_pose_ori @ np.array([0, 0, 1])
        angles = np.arccos(
            np.dot(normals, -actual_pose_vec)
        )  # use negative to get angle between normal and viewing dir
        faces_idxs_within_angle = np.where(angles <= np.radians(self.observation_config.max_incidence_angle))[0]

        # discount faces outside of sensor far plane
        faces_within_angle = detected_faces[0][faces_idxs_within_angle]
        # self.observation_counts[faces_within_angle] += 1
        if len(faces_within_angle) == 0:
            return None
        else:
            return faces_within_angle

    def get_all_detected_faces(self, pose_idx: int, pose_data: dict, sensor_data_dict: dict) -> np.ndarray:
        all_detected_faces = []
        for sensor_name, sensor_data in sensor_data_dict.items():
            detected_faces = self.get_detected_faces_from_sensor(
                pose_idx=pose_idx, pose_data=pose_data, sensor_data=sensor_data
            )
            if detected_faces is None:
                continue
            all_detected_faces.extend(detected_faces)
        if len(all_detected_faces) == 0:
            return np.array([], dtype=np.int32)
        else:
            return np.array(all_detected_faces)

    def compute_face_gains(self, faces: np.ndarray) -> np.ndarray:
        """logistic decay function for minimal return on repeated observations, with saturation at n_max observations"""
        n = self.observation_counts[faces]
        # print(f"observation counts for faces {faces}: {n}")
        # gains = self.gain_config.saturation_lambda * (1 - np.exp(-n / self.gain_config.n_max))
        gains = self.gain_config.saturation_lambda / (
            1 + np.exp(self.gain_config.steepness * (n - self.gain_config.n_midpoint))
        )
        return gains

    def compute_pose_gain(self, pose_idx: int, face_gains) -> float:
        if len(face_gains) == 0:
            return 0.0
        pose_gain = np.sum(face_gains)  # type: ignore
        return pose_gain

    def update_observations(self, pose_idx: int, detected_faces: np.ndarray) -> None:
        self.observation_counts[detected_faces] += 1
        face_gains = self.compute_face_gains(faces=detected_faces)
        pose_gain = self.compute_pose_gain(pose_idx=pose_idx, face_gains=face_gains)
        self.pose_gains[pose_idx] = pose_gain
        self.total_gain += pose_gain
        return

    def get_closest_poses(self, curr_pose):
        idxs = self.kd_tree.query_ball_point(x=curr_pose[:3], r=self.observation_config.kd_radius)
        return idxs


def plot_all_poses(sensor_extrinsics: np.ndarray, generated_poses: np.ndarray, actual_poses: np.ndarray):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=sensor_extrinsics[:, 0, 3],
            y=sensor_extrinsics[:, 1, 3],
            z=sensor_extrinsics[:, 2, 3],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="sensor extrinsics",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=generated_poses[:, 0],
            y=generated_poses[:, 1],
            z=generated_poses[:, 2],
            mode="markers",
            marker=dict(size=5, color="green"),
            name="generated poses",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=actual_poses[:, 0],
            y=actual_poses[:, 1],
            z=actual_poses[:, 2],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="actual poses",
        )
    )
    fig.show()
    return


def main():
    import threading
    from sensor_learning.plot.dash_apps import TreeCoverageApp

    path = "/home/luke/dev/pybullet/sensor-learning/data/lpy_envy_00001_20260505-174252.h5"

    with h5.File(path, "r") as file:
        data_tree = key_tree(file)

    evaluator = CoverageEvaluator(
        data_dict=data_tree,  # type: ignore
        observation_config=ObservationConfig(
            min_distance=0.1, max_distance=5.0, max_incidence_angle=45.0, kd_radius=0.08
        ),
        gain_config=GainConfig(saturation_lambda=1.5, n_midpoint=5, steepness=1.5),
    )

    app = TreeCoverageApp()
    threading.Thread(target=app.run, daemon=True).start()

    poses = data_tree["eef_poses"]["actual_poses"]  # type: ignore
    vertices = data_tree["tree"]["faces"]["vertices"]  # type: ignore
    colors = data_tree["tree"]["faces"]["color"]  # type: ignore
    sensors = data_tree["sensors"]  # type: ignore

    # rng = np.random.default_rng(seed=12)
    # # print(data_tree['eef_poses']['actual_poses'])
    # # pick a random pose to start at
    # curr_pose_idx = rng.integers(low=0, high=data_tree["eef_poses"]["actual_poses"].shape[0])  # type: ignore

    # Pick start pose at bottom of poses volume that points at origin.
    curr_pose_idx =

    # if detected_faces is None:
    #     raise ValueError("No faces detected from starting pose, try a different random seed or check data integrity.")
    all_detected_faces = []
    pose_path = []

    try:
        while True:
            # update gain and counts for current pose
            detected_faces = evaluator.get_all_detected_faces(
                pose_idx=curr_pose_idx, pose_data=poses, sensor_data_dict=sensors
            )  # type: ignore
            evaluator.update_observations(pose_idx=curr_pose_idx, detected_faces=detected_faces)
            pose_path.append(curr_pose_idx)
            all_detected_faces.extend(detected_faces)

            app.update(
                curr_pose_idx=curr_pose_idx,
                detected_faces=all_detected_faces,
                kd_idxs=evaluator.get_closest_poses(curr_pose=poses[curr_pose_idx]),
                poses=poses,
                vertices=vertices,
                colors=colors,
                observation_counts=evaluator.observation_counts,
            )

            # search for next pose to move to
            closest_idxs = evaluator.get_closest_poses(curr_pose=poses[curr_pose_idx])  # type: ignore

            # dictionary to evaluate each next potential pose
            next_step_dict = {}
            for search_idx in closest_idxs:
                for sensor_name, sensor_data in sensors.items():  # type: ignore
                    detected_faces = evaluator.get_all_detected_faces(
                        pose_idx=search_idx, pose_data=poses, sensor_data_dict=sensors
                    )  # type: ignore

                    face_gains = evaluator.compute_face_gains(faces=detected_faces)
                    pose_gain = evaluator.compute_pose_gain(pose_idx=search_idx, face_gains=face_gains)
                    next_step_dict[search_idx] = pose_gain



            # pick the next pose with the highest gain. In case of ties, pick the closest one to current pose. If still ties, pick randomly among the tied poses.
            if np.max(list(next_step_dict.values())) < 1.5:  # if max gain is very low, likely no more gain to be had from neighboring poses
                print("No more gain to be had from neighboring poses, stopping evaluation.")
                break
            max_gain = max(next_step_dict.values())
            candidates = [idx for idx, gain in next_step_dict.items() if gain == max_gain]
            if len(candidates) > 1:
                # break ties by distance to current pose
                candidate_poses = poses[candidates]  # type: ignore
                distances = np.linalg.norm(candidate_poses[:, :3] - poses[curr_pose_idx][:3], axis=1)  # type: ignore
                min_distance = np.min(distances)
                closest_candidates = [candidates[i] for i, d in enumerate(distances) if d == min_distance]
                if len(closest_candidates) > 1:
                    curr_pose_idx = rng.choice(closest_candidates)
                else:
                    curr_pose_idx = closest_candidates[0]
            else:
                curr_pose_idx = candidates[0]
            print(f"total gain so far: {evaluator.total_gain}, moving to next pose idx: {curr_pose_idx} with gain {max_gain}")

    except KeyboardInterrupt:
        print(f"Total gain accumulated: {evaluator.total_gain}")
    finally:
        import json
        output_dir = os.path.join(
            os.path.dirname(path),
            os.path.splitext(path)[0] + "_evaluation_output.json",
        )
        with open(output_dir, "w") as f:
            json.dump(
                {
                    "detected_faces": np.array(all_detected_faces).tolist(),
                    "pose_path": np.array(pose_path).tolist(),
                    "total_gain": evaluator.total_gain,
                },
                f,
                indent=4,
            )
        print(f"Saved evaluation output to {output_dir}")
    return


if __name__ == "__main__":
    main()
