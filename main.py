#!/usr/bin/env python3
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pybullet_tree_sim.utils.rotation_utils as ru
from pybullet_tree_sim.pruning_environment import PruningEnv
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.sensors.sensor_types import DataType, Modality, SensorType
from pybullet_tree_sim.tree import Tree
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
    y_range = (0.1, 0.8)
    z_range = (1.0, 1.6)
    theta_range = (0, 0)  # limit to fixed orientation for now
    phi_range = (0, 0)
    points_per_axis = 5
    angles_per_axis = 1
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

    sensors_dict = {}
    for sensor_name, sensor in robot.sensors.items():
        sensors_dict[sensor_name] = {
            "type": sensor.sensor_type,
            "data_type": sensor.data_type.value,
            "intrinsics": sensor.optical_intrinsics
            if sensor.data_type in {DataType.RGB, DataType.DEPTH, DataType.RGBD}
            else None,
            "extrinsics": [],
            "mode": {
                mode.value: {
                    "rgb": [] if mode in {Modality.RGB, Modality.DEPTH} else None,
                    "depth": [] if mode in {Modality.RGB, Modality.DEPTH} else None,
                    "pointcloud": [] if mode == Modality.POINTCLOUD else None,
                }
                for mode in sensor.modalities
            },
        }

    trial_data = {
        "trial_name": f"{tree.id_str}_{time.strftime('%Y%m%d-%H%M%S')}",
        "tree_id": tree.id_str,
        "tree_type": tree.tree_type,
        "sim_steps": np.linspace(start=0, stop=len(discrete_poses) - 1, num=len(discrete_poses), dtype=int),
        "time_step_interval": pbutils.time_step_interval,
        "sensors": sensors_dict,
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
            "actual": {
                "position": [],
                "orientation": [],
            },
        },
    }
    # logger.debug(pp.pformat(trial_data))

    for i, pose in enumerate(discrete_poses):
        pbutils.pbclient.stepSimulation()
        robot.reset_robot()
        pbutils.pbclient.stepSimulation()

        # Move robot to desired pose via IK
        joint_angles = robot.calculate_ik(position=pose[0:3], orientation=pose[3:7])
        robot.set_joint_angles_no_collision(joint_angles=joint_angles)
        pbutils.pbclient.stepSimulation()
        eef_pos, eef_ori = robot.get_current_pose(index=robot.end_effector_index)

        trial_data["eef_poses"]["actual"]["position"].append(eef_pos)
        trial_data["eef_poses"]["actual"]["orientation"].append(eef_ori)

        # Read sensors, update pyrender scene sensor poses, render
        for sensor_name, sensor in robot.sensors.items():
            for mode in sensor.modalities:
                pyb_sensor_data = sensor.read(robot=robot, pbclient=pbutils.pbclient, mode=mode)
                # logger.debug(pp.pformat(pyb_sensor_data))
                trial_data["sensors"][sensor_name]["extrinsics"].append(pyb_sensor_data["extrinsic_matrix"])
                trial_data["sensors"][sensor_name]["mode"][mode.value]["rgb"].append(
                    pyb_sensor_data["data"][0]
                ) if mode in {Modality.RGB, Modality.DEPTH} else None
                trial_data["sensors"][sensor_name]["mode"][mode.value]["depth"].append(
                    pyb_sensor_data["data"][1]
                ) if mode in {Modality.DEPTH, Modality.DEPTH} else None
                trial_data["sensors"][sensor_name]["mode"][mode.value]["pointcloud"].append(
                    pyb_sensor_data["pointcloud"]
                ) if mode == Modality.POINTCLOUD else None

    # logger.info(trial_data)
    # save pose, color, depth
    save_path = slu.save_trial_data(trial_data=trial_data)
    logger.info(f"Saved data to {save_path}")

    return


if __name__ == "__main__":
    main()
