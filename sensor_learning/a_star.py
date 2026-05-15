#!/usr/bin/env python3
"""
This script reads in h5 files containing robot poses and associated sensor data. It uses A* to maximize information gain from through a cost function of h(n) = alpha * distance_to_goal + beta * information_gain.

Then, f(n) = g(n) + h(n) where g(n) is the distance traveled so far. The script outputs a path of robot poses that maximize information gain while reaching the goal pose.
"""

import glob
import os

import h5py
import numpy as np

from pybullet_tree_sim.sensors.sensor_types import Modality


class AStarPlanner:
    def __init__(self, h5_file_path: str, alpha: float = 1.0, beta: float = 1.0):
        self.h5_file_path = h5_file_path
        self.alpha = alpha
        self.beta = beta

        self.sensors_dict = {}

        # Load data from h5 file TODO: add tree metadata (faces, etc) to file.
        with h5py.File(self.h5_file_path, "r") as f:
            self.sensors_group = f["sensors"]
            self.poses_group = f["actual_poses"]

            for sensor_name, sensor_data in self.sensors_group.items():
                self.sensors_dict[sensor_name] = {}
                for key, val in sensor_data.items():
                    if key == "extrinsics":
                        self.sensors_dict[sensor_name][key] = val[:]
                    elif key in {Modality.DEPTH, Modality.RGB}:
                        self.sensors_dict[sensor_name][Modality(key)] = {}
                        for dtype, data in val.items():
                            self.sensors_dict[sensor_name][Modality(key)][dtype] = data[:]

        return

    def heuristic(self, current_pose):
        distance_to_goal = np.linalg.norm(current_pose[:3] - self.goal_pose[:3])
        information_gain = self.get_information_gain(current_pose)
        return self.alpha * distance_to_goal - self.beta * information_gain

    def get_information_gain(self, pose):
        # Find the closest pose in the dataset and return its information gain
        distances = np.linalg.norm(self.poses[:, :3] - pose[:3], axis=1)
        closest_index = np.argmin(distances)
        return self.information_gains[closest_index]

    def plan(self):
        # Implement A* algorithm to find the optimal path
        # This is a placeholder for the actual A* implementation
        pass


def main():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(pkg_dir, "data")
    data_files = glob.glob(data_dir + "/*.h5")
    for file in data_files:
        a_star_planner = AStarPlanner(
            h5_file_path=file,
        )
    return


if __name__ == "__main__":
    main()
