#!/usr/bin/env python3

from pybullet_tree_sim.pruning_environment import PruningEnv
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.rgb_stream_visualizer import RGBStreamVisualizer
from pybullet_tree_sim.utils.trimesh_render import RenderScene

from sensor_learning.utils.pose_generator import generate_discrete_poses

import numpy as np
import time

import logging
import sensor_learning.utils.logging_conf
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug("HELLO WORLD")
logger.info("HELLO WORLD")
logger.warning("HELLO WORLD")
logger.error("HELLO WORLD")
logger.critical("HELLO WORLD")

def main():
    from scipy.spatial.transform import Rotation
    # Initialize PyBullet utils
    pbutils = PyBUtils(renders=True)

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

    discrete_poses = generate_discrete_poses(
        x_range=(-0.2, 0.2),
        y_range=(0.6, 1.6),
        z_range=(1.0, 1.6),
        theta_range=(-np.pi / 4, np.pi / 4),
        phi_range=(-np.pi / 4, np.pi / 4),
        points_per_axis=5,
        angles_per_axis=9,
    )

    # logger.info(robot.base_index)
    # logger.info(robot.end_effector_index)
    # logger.info(robot.joints)
    # import sys; sys.exit()


    for pose in discrete_poses:
        pbutils.pbclient.stepSimulation()
        # logger.debug(pose)
        # time.sleep(1)
        try:
            joint_angles = robot.calculate_ik(position=pose[0:3], orientation=pose[3:7])
            # logger.debug(joint_angles)
            robot.set_joint_angles(joint_angles=joint_angles)
            
            for i in range(100):
                pbutils.pbclient.stepSimulation()
                time.sleep(0.001)

            # log.debug(f"{robot.sensors['tof0']}")
            tof0_view_matrix = robot.get_view_mat_at_curr_pose(camera=robot.sensors["tof0"])
            tof0_rgbd = robot.get_rgbd_at_cur_pose(
                camera=robot.sensors["tof0"],
                type="sensor",
                view_matrix=tof0_view_matrix,
            )
            pbutils.pbclient.stepSimulation()
            # logger.debug(tof0_rgbd)

        except Exception as e:
            logger.error(f"{e}")
            continue

        time.sleep(0.01)



    return


if __name__ == "__main__":
    main()
