#!/usr/bin/env python3
import logging
import os
import pprint as pp

import h5py as h5
import hdf5view as h5view
import numpy as np

import sensor_learning.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_trial_data(trial_data: dict) -> str:
    """
    Save the given pose, color image, and depth image to an HDF5 file.

    trial_000.h5
    |-- tree/
    |   |-- tree_id
    |   |-- tree_type
    |   |-- faces/
    |   |   |-- cylinder_id
    |   |   |-- face_id
    |   |   |-- id_a
    |   |   |-- color
    |   |   |-- vertices
    |   |   |-- normal
    |   |   |-- t_val
    |   |   |-- theta
    |   |-- cylinders/
    |   |   |-- cylinder_id
    |   |   |-- centroid
    |   |   |-- color
    |   |   |-- length
    |   |   |-- limb_id
    |   |   |-- limb_name
    |   |   |-- orientation
    |   |   |-- radius
    |   |   |-- rot_mat
    |-- generated_poses/
    |   |-- x_range
    |   |-- y_range
    |   |-- z_range
    |   |-- theta_range
    |   |-- phi_range
    |   |-- points_per_axis
    |   |-- angles_per_axis
    |   |-- start_orientation
    |   |-- start_position
    |   |-- poses       # (N, 7) - generated end-effector poses (position + orientation)
    |-- actual_poses/
    |   |-- positions       # (N, 3) - actual end-effector positions from PyBullet sim
    |   |-- orientations    # (N, 4) - actual end-effector orientations from PyBullet sim (quaternions)
    ├── timesteps       # (15,) - sim step index or actual time in seconds
    ├── poses/
    │   ├── positions       # (15, 3)
    │   └── orientations    # (15, 4)
    └── sensors/
        └── <sensor_name>/
            ├── extrinsics  # (15, 4, 4)
            └── depth       # (15,)

    :param trial_data: A dictionary containing the trial data to be saved.
    :return: The file path of the saved HDF5 file.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(pkg_dir, "data")
    file_path = os.path.join(data_dir, trial_data["trial_name"] + ".h5")
    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir)
    if os.path.exists(file_path):
        logger.warning(f"File {file_path} already exists. It will be overwritten.")

    with h5.File(file_path, "w") as file:
        file.attrs["trial_name"] = trial_data["trial_name"]
        tree_group = file.create_group("tree")
        tree_group.attrs["tree_id"] = trial_data["tree"]["tree_id"]
        tree_group.attrs["tree_type"] = trial_data["tree"]["tree_type"]

        # Faces
        faces_group = tree_group.create_group("faces")
        faces_group.create_dataset("cylinder_id", data=trial_data["tree"]["faces"]["cylinder_id"], compression="gzip")
        faces_group.create_dataset("face_id", data=trial_data["tree"]["faces"]["id"], compression="gzip")
        faces_group.create_dataset("id_a", data=trial_data["tree"]["faces"]["id_a"], compression="gzip")
        faces_group.create_dataset("color", data=trial_data["tree"]["faces"]["color"], compression="gzip")
        faces_group.create_dataset("vertices", data=trial_data["tree"]["faces"]["vertices"], compression="gzip")
        faces_group.create_dataset("normal", data=trial_data["tree"]["faces"]["normal"], compression="gzip")
        faces_group.create_dataset("t_val", data=trial_data["tree"]["faces"]["t_val"], compression="gzip")
        faces_group.create_dataset("theta", data=trial_data["tree"]["faces"]["theta"], compression="gzip")

        # Cylinders
        cylinders_group = tree_group.create_group("cylinders")
        cylinders_group.create_dataset(
            "cylinder_id", data=trial_data["tree"]["cylinders"]["cylinder_id"], compression="gzip"
        )
        cylinders_group.create_dataset("centroid", data=trial_data["tree"]["cylinders"]["centroid"], compression="gzip")
        cylinders_group.create_dataset("color", data=trial_data["tree"]["cylinders"]["color"], compression="gzip")
        cylinders_group.create_dataset("length", data=trial_data["tree"]["cylinders"]["length"], compression="gzip")
        cylinders_group.create_dataset("limb_id", data=trial_data["tree"]["cylinders"]["limb_id"], compression="gzip")
        cylinders_group.create_dataset(
            "limb_name", data=trial_data["tree"]["cylinders"]["limb_name"].tolist(), compression="gzip"
        )
        cylinders_group.create_dataset(
            "orientation", data=trial_data["tree"]["cylinders"]["orientation"], compression="gzip"
        )
        cylinders_group.create_dataset("radius", data=trial_data["tree"]["cylinders"]["radius"], compression="gzip")
        cylinders_group.create_dataset("rot_mat", data=trial_data["tree"]["cylinders"]["rot_mat"], compression="gzip")

        # Limbs
        limbs_group = tree_group.create_group("limbs")
        children = trial_data["tree"]["limbs"]["children"]
        children_arr = np.array([np.array(c, dtype=h5.string_dtype()) for c in children], dtype=object)
        limbs_group.create_dataset(
            "children", data=children_arr, compression="gzip", dtype=h5.vlen_dtype(h5.string_dtype())
        )
        cylinder_ids = np.array(
            [
                np.array([cyl["cylinder_id"] for cyl in limb], dtype=np.int32)
                for limb in trial_data["tree"]["limbs"]["cylinders"]
            ],
            dtype=object,
        )
        limbs_group.create_dataset("cylinder_ids", data=cylinder_ids, compression="gzip", dtype=h5.vlen_dtype(np.int32))
        limbs_group.create_dataset("start_point", data=trial_data["tree"]["limbs"]["start_point"], compression="gzip")
        limbs_group.create_dataset("end_point", data=trial_data["tree"]["limbs"]["end_point"], compression="gzip")
        limbs_group.create_dataset("limb_id", data=trial_data["tree"]["limbs"]["limb_id"], compression="gzip")
        limbs_group.create_dataset("name", data=trial_data["tree"]["limbs"]["name"], compression="gzip")

        # Generated poses
        generated_poses_group = file.create_group("generated_poses")
        generated_poses_group.attrs["x_range"] = trial_data["eef_poses"]["generated"]["x_range"]
        generated_poses_group.attrs["y_range"] = trial_data["eef_poses"]["generated"]["y_range"]
        generated_poses_group.attrs["z_range"] = trial_data["eef_poses"]["generated"]["z_range"]
        generated_poses_group.attrs["theta_range"] = trial_data["eef_poses"]["generated"]["theta_range"]
        generated_poses_group.attrs["phi_range"] = trial_data["eef_poses"]["generated"]["phi_range"]
        generated_poses_group.attrs["points_per_axis"] = trial_data["eef_poses"]["generated"]["points_per_axis"]
        generated_poses_group.attrs["angles_per_axis"] = trial_data["eef_poses"]["generated"]["angles_per_axis"]
        generated_poses_group.attrs["start_orientation"] = trial_data["eef_poses"]["generated"]["start_orientation"]
        generated_poses_group.attrs["start_position"] = trial_data["eef_poses"]["generated"]["start_position"]
        generated_poses_group.create_dataset(
            "poses", data=trial_data["eef_poses"]["generated"]["poses"], compression="gzip"
        )

        # Actual poses (from PyBullet sim)
        actual_poses_group = file.create_group("actual_poses")
        actual_poses_group.create_dataset(
            "positions", data=trial_data["eef_poses"]["actual"]["position"], compression="gzip"
        )
        actual_poses_group.create_dataset(
            "orientations", data=trial_data["eef_poses"]["actual"]["orientation"], compression="gzip"
        )

        simulation_steps_group = file.create_group("simulation_steps")
        simulation_steps_group.attrs["time_step_interval"] = trial_data["time_step_interval"]
        simulation_steps_group.create_dataset(
            "steps", data=trial_data["sim_steps"], compression="gzip"
        )  # default timestep is 1/240

        sensors_group = file.create_group("sensors")
        for sensor_name, sensor_data in trial_data["sensors"].items():
            sensor_group = sensors_group.create_group(sensor_name)
            sensor_group.create_dataset("extrinsics", data=sensor_data["extrinsics"], compression="gzip")
            for mode, mode_data in sensor_data["mode"].items():
                mode_group = sensor_group.create_group(mode)
                if mode_data["rgb"] is not None:
                    mode_group.create_dataset("rgb", data=mode_data["rgb"], compression="gzip")
                if mode_data["depth"] is not None:
                    mode_group.create_dataset("depth", data=mode_data["depth"], compression="gzip")
                if mode_data["pointcloud"] is not None:
                    mode_group.create_dataset("pointcloud", data=mode_data["pointcloud"], compression="gzip")

    return file_path
