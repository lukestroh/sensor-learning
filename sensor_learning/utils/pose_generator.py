#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet_tree_sim.utils.rotation_utils as ru


def generate_discrete_poses(
    x_range,
    y_range,
    z_range,
    theta_range,
    phi_range,
    points_per_axis,
    angles_per_axis,
    start_orientation=np.array([0, 0, 1]),
):

    points = generate_discrete_pts(x_range, y_range, z_range, size=points_per_axis)
    orientations = generate_uniform_spherical_pts(
        theta_range=theta_range, phi_range=phi_range, size=angles_per_axis, start_orientation=start_orientation
    )
    quats = ru.ori_vec_to_quat(orientations)

    points_repeated = np.repeat(points, quats.shape[0], axis=0)
    quats_tiled = np.tile(quats, (points.shape[0], 1))
    poses = np.hstack((points_repeated, quats_tiled))

    # Return only unique poses
    poses = np.unique(poses, axis=0)
    return poses


def generate_discrete_pts(x_range, y_range, z_range, size):
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    z = np.linspace(z_range[0], z_range[1], size)

    grid = np.meshgrid(x, y, z, indexing="ij")
    discrete_pts = np.vstack(list(map(np.ravel, grid))).T
    return discrete_pts


def generate_uniform_spherical_pts(theta_range, phi_range, size, start_orientation=np.array([0, 0, 1])):
    theta = np.linspace(theta_range[0], theta_range[1], size)
    phi = np.linspace(phi_range[0], phi_range[1], size)

    grid = np.meshgrid(theta, phi, indexing="ij")
    spherical_pts = np.vstack(list(map(np.ravel, grid))).T

    vectors = np.empty((spherical_pts.shape[0], 3), dtype=np.float64)

    vectors[:, 0] = np.sin(spherical_pts[:, 0]) * np.cos(spherical_pts[:, 1])
    vectors[:, 1] = np.sin(spherical_pts[:, 0]) * np.sin(spherical_pts[:, 1])
    vectors[:, 2] = np.cos(spherical_pts[:, 0])

    # Align generated vectors with start_orientation vector
    rotation_axis = np.cross(np.array([0, 0, 1]), start_orientation)
    rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), start_orientation) / (np.linalg.norm(start_orientation)))
    rot_vecs = rotation_axis * rotation_angle
    rot_mats = Rotation.from_rotvec(rot_vecs).as_matrix()
    vectors = (rot_mats @ vectors.T).T

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


if __name__ == "__main__":
    import pprint as pp
    import sensor_learning.utils.plot as slp

    x_range = (-0.5, 0.5)
    y_range = (-0.5, 0.5)
    z_range = (0.0, 1.0)
    point_size = 5

    theta_range = (-np.pi / 4, np.pi / 4)
    phi_range = (-np.pi / 2, np.pi / 2)
    angular_size = 9

    # poses_q, poses_v = generate_discrete_poses(x_range, y_range, z_range, roll_range, pitch_range, yaw_range, size)
    points = generate_discrete_pts(x_range, y_range, z_range, point_size)
    orientations = generate_uniform_spherical_pts(theta_range, phi_range, angular_size)
    quats = ru.ori_vec_to_quat(orientations)

    # slp.plot_quaternions(quats).show()

    # slp.plot_vectors(orientations).show()

    discrete_poses = generate_discrete_poses(
        x_range=(-0.2, 0.2),
        y_range=(0.6, 1.0),
        z_range=(1.0, 1.6),
        theta_range=(-np.pi / 4, np.pi / 4),
        phi_range=(-np.pi / 4, np.pi / 4),
        points_per_axis=5,
        angles_per_axis=5,
    )
    unique_poses = np.unique(discrete_poses, axis=0)
    print(f"Generated {discrete_poses.shape[0]} poses, {unique_poses.shape[0]} unique poses.")

    slp.plot_quaternions(unique_poses[:, 3:7]).show()
