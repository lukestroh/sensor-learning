#!/usr/bin/env python3

from pybullet_tree_sim.pruning_environment import PruningEnv
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.trimesh_render import RenderScene

import numpy as np
from scipy.spatial.transform import Rotation


import time
from zenlog import log

from pybullet_tree_sim.utils.pyb_utils import PyBUtils


def load_scene():
    pbutils = PyBUtils(renders=True)
    robot_start_orientation = Rotation.from_euler("xyz", [0, 0, 180], degrees=True).as_quat()
    robot = Robot(
        pbclient=pbutils.pbclient,
        position=[0, 1, 0],
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

    penv = PruningEnv(
        pbutils=pbutils,
        verbose=True,
    )
    tree = Tree(pbutils=pbutils, tree_id=0, tree_type="envy", namespace="lpy")
    penv.activate_tree(tree=tree, include_support_posts=False)

    return set(robot, tree, penv)
