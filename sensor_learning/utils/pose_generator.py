#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation

def generate_discrete_poses(x_range, y_range, z_range, theta_range, phi_range, points_per_axis, angles_per_axis):
    
    points = generate_discrete_pts(x_range, y_range, z_range, points_per_axis)
    orientations = generate_uniform_spherical_pts(theta_range, phi_range, angles_per_axis)
    quats = ori_vec_to_quat(orientations)

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

    grid = np.meshgrid(x, y, z, indexing='ij')
    discrete_pts = np.vstack(list(map(np.ravel, grid))).T
    return discrete_pts

def generate_uniform_spherical_pts(theta_range, phi_range, size):
    theta = np.linspace(theta_range[0], theta_range[1], size)
    phi = np.linspace(phi_range[0], phi_range[1], size)

    grid = np.meshgrid(theta, phi, indexing='ij')
    spherical_pts = np.vstack(list(map(np.ravel, grid))).T

    vectors = np.empty((spherical_pts.shape[0], 3), dtype=np.float64)

    vectors[:, 0] = np.sin(spherical_pts[:, 0]) * np.cos(spherical_pts[:, 1])
    vectors[:, 1] = np.sin(spherical_pts[:, 0]) * np.sin(spherical_pts[:, 1])
    vectors[:, 2] = np.cos(spherical_pts[:, 0])

    default_view_vec = np.array([0, -1, 0])
    # Align generated vectors with default view vector
    rotation_axis = np.cross(np.array([0, 0, 1]), default_view_vec)
    rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), default_view_vec) / (np.linalg.norm(default_view_vec)))
    rot_vecs = rotation_axis * rotation_angle
    rot_mats = Rotation.from_rotvec(rot_vecs).as_matrix()
    vectors = (rot_mats @ vectors.T).T

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def ori_vec_to_quat(ori_vecs: np.ndarray) -> np.ndarray:
    """Convert orientation vectors to quaternions

    :param ori_vecs: Orientation vectors
    :type ori_vecs: np.ndarray
    :return: Quaternions
    :rtype: np.ndarray
    """
    norms = np.linalg.norm(ori_vecs, axis=1, keepdims=True)
    ori_vecs = ori_vecs / norms
    world_z = np.array([0, 0, 1])
    angles = np.arccos(np.dot(ori_vecs, world_z) / (np.linalg.norm(ori_vecs, axis=1) * np.linalg.norm(world_z)))
    rot_vecs = np.cross(world_z, ori_vecs)
    rot_vec_norms = np.linalg.norm(rot_vecs, axis=1, keepdims=True)
    rot_vec_norms = np.divide(rot_vecs, rot_vec_norms, where=(rot_vec_norms > 1e-10), out=np.zeros_like(rot_vecs))
    rot_vecs = rot_vec_norms * angles[:, np.newaxis]
    quats = Rotation.from_rotvec(rot_vecs).as_quat()
    return quats


if __name__ == "__main__":
    import pprint as pp
    import sensor_learning.utils.plot as slp

    x_range = (-0.5, 0.5)
    y_range = (-0.5, 0.5)
    z_range = (0.0, 1.0)
    point_size = 5

    theta_range = (-np.pi/4, np.pi/4)
    phi_range = (-np.pi/2, np.pi/2)
    angular_size = 9

    # poses_q, poses_v = generate_discrete_poses(x_range, y_range, z_range, roll_range, pitch_range, yaw_range, size)
    points = generate_discrete_pts(x_range, y_range, z_range, point_size)
    orientations = generate_uniform_spherical_pts(theta_range, phi_range, angular_size)
    quats = ori_vec_to_quat(orientations)

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