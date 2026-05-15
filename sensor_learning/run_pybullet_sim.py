#!/usr/bin/env python3
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pybullet_tree_sim.utils.rotation_utils as ru
import tqdm
from pybullet_tree_sim.pruning_environment import PruningEnv
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.sensors.sensor_types import DataType, Modality, SensorType
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.tree_metadata import Cylinder
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.rgb_stream_visualizer import RGBStreamVisualizer
from pybullet_tree_sim.utils.trimesh_render import RenderScene

import sensor_learning.utils.logging_conf
import sensor_learning.utils.pose_generator as pg
import sensor_learning.utils.save_data_utils as slu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import os

pkg_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    import pprint as pp

    from scipy.spatial.transform import Rotation

    # Initialize PyBullet utils
    pbutils = PyBUtils(renders=False)

    # Create robot
    robot_start_orientation = Rotation.from_euler("xyz", [0, 0, 180], degrees=True).as_quat()
    robot = Robot(
        pbclient=pbutils.pbclient,
        position=[0, 1.5, 0],
        orientation=robot_start_orientation,
        init_joint_angles=(
            0,  # linear slider
            -np.pi / 2 + np.pi / 4,
            -np.pi * 2 / 3,
            np.pi * 2 / 3,
            -np.pi,
            -np.pi / 2,
            0,
        ),
    )

    # Create pruning environment
    penv = PruningEnv(pbutils=pbutils, name="pruning_env", verbose=True)

    tree = Tree(
        pbutils=pbutils,
        meshes_root="/home/luke/dev/pybullet/pybullet-tree-sim/pybullet_tree_sim/meshes",
        tree_id=1,
        tree_type="envy",
        namespace="LPy",
    )
    penv.activate_tree(tree=tree, include_support_posts=False)

    # # Run the sim a little just to get the environment properly loaded.
    for i in range(100):
        pbutils.pbclient.stepSimulation()
        time.sleep(0.01)

    # Create the PyRender scene, add sensors. Tree gets recolored in Tree() instantiation and exported to obj path
    pyr_scene = RenderScene(mesh=tree.recolored_mesh)
    for sensor_name, sensor in robot.sensors.items():
        pyb_sensor_data = sensor.read(robot=robot, pbclient=pbutils.pbclient)
        # check sensor modality, if optical sensor, add to pyrender scene
        if sensor.data_type in {DataType.RGB, DataType.DEPTH, DataType.RGBD}:
            sensor_extrinsics = np.asarray(pyb_sensor_data["extrinsic_matrix"]).reshape((4, 4), order="F")
            sensor_pose = np.linalg.inv(sensor_extrinsics)
            pyr_scene.add_optical_sensor(optical_sensor=sensor, pose=sensor_pose, sensor_name=sensor_name)
        if sensor.data_type in {DataType.POINTCLOUD}:
            ...

    # Generate discrete poses within a workspace volume
    start_pos, start_ori = robot.get_current_pose(index=robot.end_effector_index)  # TODO: change arm eef
    start_ori = ru.quat_to_ori_vec(start_ori)
    x_range = (-0.2, 0.2)  # these poses are relative to the robot base
    y_range = (0.35, 0.75)
    z_range = (1.1, 1.5)
    theta_range = (-np.pi / 4, np.pi / 4)  # limit to fixed orientation for now
    phi_range = (-np.pi / 4, np.pi / 4)
    points_per_axis = 10
    angles_per_axis = 3
    discrete_poses = pg.generate_discrete_poses(
        x_range=x_range,  # these poses are relative to the robot base
        y_range=y_range,
        z_range=z_range,
        theta_range=theta_range,
        phi_range=phi_range,
        points_per_axis=points_per_axis,
        angles_per_axis=angles_per_axis,
        start_orientation=start_ori,
    )
    print(start_pos, start_ori)

    # Faces
    faces_keys = ["color", "cylinder_id", "id", "id_a", "normal", "t_val", "theta", "vertices"]
    faces = [f.to_dict() for f in tree.faces]
    zipped_faces = zip(*[[face[key] for key in faces_keys] for face in faces])
    dtypes = (np.int32, np.int32, np.int32, np.int32, np.float32, np.float32, np.float32, np.float32)
    faces_dict = {k: np.array(list(v), dtype=d) for k, v, d in zip(faces_keys, zipped_faces, dtypes)}

    # Cylinders
    cylinder_keys = [
        "centroid",
        "color",
        "cylinder_id",
        "length",
        "limb_id",
        "limb_name",
        "orientation",
        "radius",
        "rot_mat",
    ]
    cylinders = [cyl.to_dict() for cyl in tree.cylinders]
    zipped_cylinders = zip(*[[cyl[key] for key in cylinder_keys] for cyl in cylinders])
    dtypes = (np.float32, np.int32, np.int32, np.float32, np.int32, np.str_, np.float32, np.float32, np.float32)
    cylinders_dict = {k: np.array(list(v), dtype=d) for k, v, d in zip(cylinder_keys, zipped_cylinders, dtypes)}

    # Limbs
    limb_keys = ["children", "cylinders", "start_point", "end_point", "limb_id", "name"]
    limbs = [limb.to_dict() for limb in tree.limbs.values()]
    limbs_dict = {k: [limb[k] for limb in limbs] for k in limb_keys}

    # Sensors
    sensors_metadata_dict = {}
    for sensor_name, sensor in robot.sensors.items():
        sensors_metadata_dict[sensor_name] = {
            "n_poses": len(discrete_poses),
            "type": sensor.sensor_type,
            "data_type": sensor.data_type.value,
            "intrinsics": (
                sensor.optical_intrinsics if sensor.data_type in {DataType.RGB, DataType.DEPTH, DataType.RGBD} else None
            ),
        }
        # print(sensor.optical_intrinsics)
        # import sys; sys.exit()

    trial_metadata = {
        "trial_name": f"{tree.id_str}_{time.strftime('%Y%m%d-%H%M%S')}",
        "sim_steps": np.linspace(start=0, stop=len(discrete_poses) - 1, num=len(discrete_poses), dtype=int),
        "time_step_interval": pbutils.time_step_interval,
        # "sensors": sensors_dict,
        "eef_poses": {
            "generated": {
                "poses": discrete_poses,
                "x_range": x_range,
                "y_range": y_range,
                "z_range": z_range,
                "theta_range": theta_range,
                "phi_range": phi_range,
                "points_per_axis": points_per_axis,
                "angles_per_axis": angles_per_axis,
                "start_position": start_pos,
                "start_orientation": start_ori,
            },
        },
    }

    tree_metadata = {
        "tree_id": tree.id_str,
        "tree_type": tree.tree_type,
        "faces": faces_dict,
        "cylinders": cylinders_dict,
        "limbs": limbs_dict,
    }

    # set up h5 file, save metadata
    file_path = slu.save_trial_metadata(trial_metadata=trial_metadata)
    file_path = slu.save_tree_metadata(tree_metadata=tree_metadata, file_path=file_path)
    file_path = slu.initialize_sensor_data(sensor_metadata=sensors_metadata_dict, file_path=file_path)

    def reset_trial_data():
        return {
            "index": [],
            "eef_poses": {"poses": []},
            "sensors": {
                sensor_name: {
                    "extrinsics": [],
                    "mode": {
                        mode.value: {"rgb": [], "depth": {"depth": [], "depth_rgb": []}, "pointcloud": []}
                        for mode in sensor.modalities
                    },
                }
                for sensor_name, sensor in robot.sensors.items()
            },
        }

    trial_data = reset_trial_data()
    for i, pose in enumerate(tqdm.tqdm(discrete_poses)):
        # Batch save data to h5 to avoid memory issues
        if i % 100 == 0 and i > 0:
            file_path = slu.save_trial_data(trial_data=trial_data, file_path=file_path)
            trial_data = reset_trial_data()
        # if i % 1000 == 0 and i > 0:
        #     pbutils.pbclient.resetSimulation()
        #     robot = Robot(
        #         pbclient=pbutils.pbclient,
        #         position=[0, 1.5, 0],
        #         orientation=robot_start_orientation,
        #         init_joint_angles=(
        #             0,  # linear slider
        #             -np.pi / 2 + np.pi / 4,
        #             -np.pi * 2 / 3,
        #             np.pi * 2 / 3,
        #             -np.pi,
        #             -np.pi / 2,
        #             0,
        #         ),
        #     )
        #     tree = Tree(
        #         pbutils=pbutils,
        #         meshes_root="/home/luke/dev/pybullet/pybullet-tree-sim/pybullet_tree_sim/meshes",
        #         tree_id=1,
        #         tree_type="envy",
        #         namespace="LPy",
        #     )
        #     penv.activate_tree(tree=tree, include_support_posts=False)
        #     pbutils.pbclient.stepSimulation()

        robot.reset_robot()
        pbutils.pbclient.stepSimulation()

        # Move robot to desired pose via IK
        joint_angles = robot.calculate_ik(position=pose[0:3], orientation=pose[3:7])
        robot.set_joint_angles_no_collision(joint_angles=joint_angles)
        pbutils.pbclient.stepSimulation()
        eef_pos, eef_ori = robot.get_current_pose(index=robot.end_effector_index)
        trial_data["eef_poses"]["poses"].append(np.concatenate((eef_pos, eef_ori)))
        trial_data["index"].append(i)

        # print(robot.end_effector_index)
        # pp.pprint(robot.links)

        # base_pos, base_ori = robot.get_current_pose(robot.base_index)
        # print(f"Base pos: {base_pos}")

        # Read sensors, update pyrender scene sensor poses, render
        for sensor_name, sensor in robot.sensors.items():
            for mode in sensor.modalities:
                pyb_sensor_data = sensor.read(robot=robot, pbclient=pbutils.pbclient, mode=mode)
                sensor_extrinsics = np.asarray(pyb_sensor_data["extrinsic_matrix"]).reshape((4, 4), order="F")
                pyr_scene.update_sensor_pose(sensor=sensor, pose=np.linalg.inv(sensor_extrinsics))
                pyr_sensor_data = pyr_scene.render_optical_sensor(sensor=sensor)
                
                # RGB
                trial_data["sensors"][sensor_name]["extrinsics"].append(np.linalg.inv(sensor_extrinsics))
                (
                    trial_data["sensors"][sensor_name]["mode"][mode.value]["rgb"].append(
                        pyr_sensor_data[mode.value]["rgb"]
                    )
                    if mode in {Modality.RGB}
                    else None
                )
                # Depth
                (
                    trial_data["sensors"][sensor_name]["mode"][mode.value]["depth"]["depth"].append(
                        pyb_sensor_data["data"][1]
                    )
                    if mode in {Modality.DEPTH}
                    else None
                )
                (
                    trial_data["sensors"][sensor_name]["mode"][mode.value]["depth"]["depth_rgb"].append(
                        pyr_sensor_data[mode.value]["rgb"]
                    )
                    if mode in {Modality.DEPTH}
                    else None
                )
                # Pointcloud
                (
                    trial_data["sensors"][sensor_name]["mode"][mode.value]["pointcloud"].append(
                        pyb_sensor_data["pointcloud"]
                    )
                    if mode == Modality.POINTCLOUD
                    else None
                )

    file_path = slu.save_trial_data(trial_data=trial_data, file_path=file_path)
    logger.info(f"Saved data to {file_path}")

    return


if __name__ == "__main__":
    main()
