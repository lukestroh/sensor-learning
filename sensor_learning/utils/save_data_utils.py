#!/usr/bin/env python3
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

import logging
import os
import pprint as pp

import h5py as h5
import hdf5view as h5view
import numpy as np
from pybullet_tree_sim.sensors.sensor_types import Modality

import sensor_learning.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TARGET_CHUNK_SIZE = 512 * 1024  # 512 KB


def get_file_path(trial_metadata: dict):
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(pkg_dir, "data")
    file_path = os.path.join(data_dir, trial_metadata["trial_name"] + ".h5")
    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir)
    if os.path.exists(file_path):
        logger.warning(f"File {file_path} already exists. It will be overwritten.")
    return file_path


def save_trial_metadata(trial_metadata: dict) -> str:
    file_path = get_file_path(trial_metadata)
    with h5.File(file_path, "a") as file:
        file.attrs["trial_name"] = trial_metadata["trial_name"]

        # Eef poses
        eef_poses_group = file.create_group("eef_poses")
        # Generated poses
        generated_poses_dset = eef_poses_group.create_dataset(
            "generated_poses", data=trial_metadata["eef_poses"]["generated"]["poses"], compression="gzip"
        )
        generated_poses_dset.attrs["x_range"] = trial_metadata["eef_poses"]["generated"]["x_range"]
        generated_poses_dset.attrs["y_range"] = trial_metadata["eef_poses"]["generated"]["y_range"]
        generated_poses_dset.attrs["z_range"] = trial_metadata["eef_poses"]["generated"]["z_range"]
        generated_poses_dset.attrs["theta_range"] = trial_metadata["eef_poses"]["generated"]["theta_range"]
        generated_poses_dset.attrs["phi_range"] = trial_metadata["eef_poses"]["generated"]["phi_range"]
        generated_poses_dset.attrs["points_per_axis"] = trial_metadata["eef_poses"]["generated"]["points_per_axis"]
        generated_poses_dset.attrs["angles_per_axis"] = trial_metadata["eef_poses"]["generated"]["angles_per_axis"]
        generated_poses_dset.attrs["start_orientation"] = trial_metadata["eef_poses"]["generated"]["start_orientation"]
        generated_poses_dset.attrs["start_position"] = trial_metadata["eef_poses"]["generated"]["start_position"]
        
        # Actual poses
        n_poses = len(trial_metadata["eef_poses"]["generated"]["poses"])
        bytes_per_entry = trial_metadata['eef_poses']['generated']['poses'].dtype.itemsize * 7
        entries_per_chunk = min(n_poses, TARGET_CHUNK_SIZE // bytes_per_entry)
        actual_poses_dset = eef_poses_group.create_dataset(
            name="actual_poses",
            shape=(n_poses, 7),
            dtype=np.float32,
            chunks=(entries_per_chunk, 7),  # chunk size for efficient appending
            compression="gzip",
        )
        

        simulation_steps_group = file.create_group("simulation_steps")
        simulation_steps_group.attrs["time_step_interval"] = trial_metadata["time_step_interval"]
        simulation_steps_group.create_dataset(
            "steps", data=trial_metadata["sim_steps"], compression="gzip"
        )  # default timestep is 1/240

    return file_path


def save_tree_metadata(tree_metadata: dict, file_path: str) -> str:
    """
    Save the given tree data to an HDF5 file.

    :param tree_metadata: A dictionary containing the tree data to be saved.
    :param file_path: The file path of the HDF5 file to save the data to.
    """
    with h5.File(file_path, "a") as file:
        tree_group = file.create_group("tree")
        tree_group.attrs["tree_id"] = tree_metadata["tree_id"]
        tree_group.attrs["tree_type"] = tree_metadata["tree_type"]

        # Faces
        faces_group = tree_group.create_group("faces")
        faces_group.create_dataset("cylinder_id", data=tree_metadata["faces"]["cylinder_id"], compression="gzip")
        faces_group.create_dataset("face_id", data=tree_metadata["faces"]["id"], compression="gzip")
        faces_group.create_dataset("id_a", data=tree_metadata["faces"]["id_a"], compression="gzip")
        faces_group.create_dataset("color", data=tree_metadata["faces"]["color"], compression="gzip")
        n_vertices = len(tree_metadata['faces']['vertices'])
        bytes_per_entry = tree_metadata['faces']['vertices'].dtype.itemsize * 9  # 3 vertices with 3 coordinates each
        entries_per_chunk = min(n_vertices, TARGET_CHUNK_SIZE // bytes_per_entry)
        faces_group.create_dataset(
            "vertices", data=tree_metadata["faces"]["vertices"], chunks=(entries_per_chunk, 3, 3), compression="gzip"
        )
        faces_group.create_dataset("normal", data=tree_metadata["faces"]["normal"], compression="gzip")
        faces_group.create_dataset("t_val", data=tree_metadata["faces"]["t_val"], compression="gzip")
        faces_group.create_dataset("theta", data=tree_metadata["faces"]["theta"], compression="gzip")

        # Cylinders
        cylinders_group = tree_group.create_group("cylinders")
        cylinders_group.create_dataset(
            "cylinder_id", data=tree_metadata["cylinders"]["cylinder_id"], compression="gzip"
        )
        cylinders_group.create_dataset("centroid", data=tree_metadata["cylinders"]["centroid"], compression="gzip")
        cylinders_group.create_dataset("color", data=tree_metadata["cylinders"]["color"], compression="gzip")
        cylinders_group.create_dataset("length", data=tree_metadata["cylinders"]["length"], compression="gzip")
        cylinders_group.create_dataset("limb_id", data=tree_metadata["cylinders"]["limb_id"], compression="gzip")
        cylinders_group.create_dataset(
            "limb_name", data=tree_metadata["cylinders"]["limb_name"].tolist(), compression="gzip"
        )
        cylinders_group.create_dataset(
            "orientation", data=tree_metadata["cylinders"]["orientation"], compression="gzip"
        )
        cylinders_group.create_dataset("radius", data=tree_metadata["cylinders"]["radius"], compression="gzip")
        n_cyls = len(tree_metadata["cylinders"]["cylinder_id"])
        bytes_per_entry = tree_metadata["cylinders"]["rot_mat"].dtype.itemsize * 9  # 3x3 matrix
        entries_per_chunk = min(n_cyls, TARGET_CHUNK_SIZE // bytes_per_entry)
        cylinders_group.create_dataset(
            "rot_mat", data=tree_metadata["cylinders"]["rot_mat"], chunks=(entries_per_chunk, 3, 3), compression="gzip"
        )

        # Limbs
        limbs_group = tree_group.create_group("limbs")
        children = tree_metadata["limbs"]["children"]
        children_arr = np.array([np.array(c, dtype=h5.string_dtype()) for c in children], dtype=object)
        limbs_group.create_dataset(
            "children", data=children_arr, compression="gzip", dtype=h5.vlen_dtype(h5.string_dtype())
        )
        cylinder_ids = np.array(
            [
                np.array([cyl["cylinder_id"] for cyl in limb], dtype=np.int32)
                for limb in tree_metadata["limbs"]["cylinders"]
            ],
            dtype=object,
        )
        limbs_group.create_dataset("cylinder_ids", data=cylinder_ids, compression="gzip", dtype=h5.vlen_dtype(np.int32))
        limbs_group.create_dataset("start_point", data=tree_metadata["limbs"]["start_point"], compression="gzip")
        limbs_group.create_dataset("end_point", data=tree_metadata["limbs"]["end_point"], compression="gzip")
        limbs_group.create_dataset("limb_id", data=tree_metadata["limbs"]["limb_id"], compression="gzip")
        limbs_group.create_dataset("name", data=tree_metadata["limbs"]["name"], compression="gzip")

    return file_path


def initialize_sensor_data(sensor_metadata: dict, file_path: str) -> str:
    with h5.File(file_path, "a") as file:
        sensors_group = file.create_group("sensors")
        # Save metadata
        for sensor_name, sensor_info in sensor_metadata.items():
            sensor_group = sensors_group.create_group(sensor_name)
            sensor_group.attrs["name"] = sensor_name
            sensor_group.attrs["type"] = sensor_info["type"]
            sensor_group.attrs["data_type"] = sensor_info["data_type"]
            # print(sensor_info)
            # 

            bytes_per_entry = np.empty((4, 4), dtype=np.float32).itemsize * 16  # 4x4 matrix
            entries_per_chunk = min(sensor_info['n_poses'], TARGET_CHUNK_SIZE // bytes_per_entry)

            sensor_group.create_dataset(
                name="extrinsics",
                shape=(sensor_info["n_poses"], 4, 4),
                dtype=np.float32,
                chunks=(entries_per_chunk, 4, 4),  # chunk size for efficient appending
                compression="gzip",
                compression_opts=4,
            )
            for mode_type, mode_intrinsics in sensor_info["intrinsics"].items():
                mode_group = sensor_group.create_group(mode_type)
                mode_group.attrs["fx"] = mode_intrinsics["fx"]
                mode_group.attrs["fy"] = mode_intrinsics["fy"]
                mode_group.attrs["cx"] = mode_intrinsics["cx"]
                mode_group.attrs["cy"] = mode_intrinsics["cy"]
                mode_group.attrs["width"] = mode_intrinsics["width"]
                mode_group.attrs["height"] = mode_intrinsics["height"]
                mode_group.attrs["znear"] = mode_intrinsics["znear"]
                mode_group.attrs["zfar"] = mode_intrinsics["zfar"]

                if mode_type == Modality.RGB:
                    mode_group.create_dataset(
                        name="rgb",
                        shape=(
                            sensor_info["n_poses"],
                            mode_intrinsics["width"],
                            mode_intrinsics["height"],
                            3,
                        ),
                        dtype=np.uint8,
                        chunks=(1, mode_intrinsics["width"], mode_intrinsics["height"], 3),
                        compression="gzip",
                    )
                if mode_type == Modality.DEPTH:
                    mode_group.create_dataset(
                        name="depth_rgb",
                        shape=(sensor_info["n_poses"], mode_intrinsics["width"], mode_intrinsics["height"], 3),
                        dtype=np.int32,
                        chunks=(1, mode_intrinsics["width"], mode_intrinsics["height"], 3),
                        compression="gzip",
                    )
                    mode_group.create_dataset(
                        name="depth",
                        shape=(
                            sensor_info["n_poses"],
                            mode_intrinsics["width"],
                            mode_intrinsics["height"],
                        ),
                        dtype=np.float32,
                        chunks=(1, mode_intrinsics["width"], mode_intrinsics["height"]),
                        compression="gzip",
                    )
                if mode_type == Modality.POINTCLOUD:
                    mode_group.create_dataset(
                        name="pointcloud",
                        shape=(sensor_info["n_poses"], ...),
                        dtype=np.float32,
                        chunks=(1, ...),  # chunk size for efficient appending
                        compression="gzip",
                    )

    return file_path


def save_trial_data(trial_data: dict, file_path: str) -> str:
    with h5.File(file_path, "a") as file:
        batch_start = trial_data["index"][0]
        batch_end = trial_data["index"][-1] + 1
        
        file['eef_poses/actual_poses'][batch_start:batch_end] = trial_data['eef_poses']['poses'] # type: ignore

        for sensor_name, sensor_data in trial_data["sensors"].items():
            file[f"sensors/{sensor_name}/extrinsics"][batch_start:batch_end] = sensor_data["extrinsics"] # type: ignore
            
            for mode, mode_data in sensor_data["mode"].items():
                mode_base = f"sensors/{sensor_name}/{mode}"
                
                if mode_data["rgb"]:
                    print(mode_data["rgb"])
                    file[f"{mode_base}/rgb"][batch_start:batch_end] = mode_data["rgb"] # type: ignore
                if mode_data["depth"]:
                    file[f"{mode_base}/depth_rgb"][batch_start:batch_end] = mode_data['depth']["depth_rgb"] # type: ignore
                    file[f"{mode_base}/depth"][batch_start:batch_end] = mode_data['depth']["depth"] # type: ignore
                if mode_data["pointcloud"]:
                    file[f"{mode_base}/pointcloud"][batch_start:batch_end] = mode_data["pointcloud"] # type: ignore

    return file_path
