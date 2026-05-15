#!/usr/bin/env python3
import os
import tempfile

import bpy
import numpy as np
from scipy.spatial.transform import Rotation


def clear_all_objects():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    return


def clear_material(material: bpy.types.Material):
    material.use_nodes = True
    if material.node_tree:
        material.node_tree.nodes.clear()
        material.node_tree.links.clear()
    return


def create_obj_from_file(filepath: str):
    bpy.ops.wm.obj_import(filepath=filepath, forward_axis="Y", up_axis="Z")
    return bpy.context.selected_objects[0]


def create_obj_with_vertex_colors(name: str, obj: bpy.types.Object, type: str):
    materials = bpy.data.materials
    mat_name = f"mat_{name}"
    material = materials.get(mat_name)
    if material:
        bpy.data.materials.remove(material)
    material = materials.new(name=mat_name)
    clear_material(material)

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    input_node = nodes.new(type="ShaderNodeVertexColor")
    if type == "emission":
        shader_node = nodes.new(type="ShaderNodeEmission")
        shader_node.inputs["Strength"].default_value = 1.0
        links.new(input_node.outputs["Color"], shader_node.inputs["Color"])
        links.new(shader_node.outputs["Emission"], output_node.inputs["Surface"])
    elif type == "diffuse":
        ...

    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    bpy.context.scene.collection.objects.link(obj)
    return


def create_camera_with_intrinsics(
    name: str, fx_mm: float, fy_mm: float, width: int, height: int, z_near: float, z_far: float
) -> bpy.types.Object:
    camera_data: bpy.types.Camera = bpy.data.cameras.new(name=name)
    camera_obj: bpy.types.Object = bpy.data.objects.new(name=name, object_data=camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj  # set as active camera

    # Intrinsics
    update_camera_intrinsics(
        camera=camera_data,
        fx_mm=fx_mm,
        fy_mm=fy_mm,
        width=width,
        height=height,
        z_near=z_near,
        z_far=z_far,
    )

    return camera_obj


def update_camera_intrinsics(
    camera: bpy.types.Camera, fx_mm: float, fy_mm: float, width: int, height: int, z_near: float, z_far: float
):
    camera.sensor_fit = "HORIZONTAL"
    camera.sensor_width = 0.1
    camera.lens = fx_mm * camera.sensor_width / width  # f_in_mm = fx_px * sensor_mm / view_fac_px

    # Encode fy != fx as a pixel aspect ratio
    # fx/fy = pixel_aspect_y/pixel_aspect_x (with pixel_aspect_x=1)
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = fx_mm / fy_mm

    camera.clip_start = z_near
    camera.clip_end = z_far

    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100

    return


def update_camera_position(camera: bpy.types.Object, extrinsics: np.ndarray):
    """Blender expects camera->world TF, and directions as looking down -Z, with +Y as up. We assume extrinsics is world->camera, with +Z forward and +Y up."""
    R_world_to_cam = extrinsics[:3, :3]
    t_world_to_cam = extrinsics[:3, 3]
    R_cam_to_world = R_world_to_cam.T
    # t_cam_to_world = -R_cam_to_world @ t_world_to_cam
    # # OpenCV → Blender axis correction
    # cv_to_blender = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # R_cam_to_world = cv_to_blender @ R_cam_to_world
    # t_cam_to_world = cv_to_blender @ t_cam_to_world
    camera.location = t_world_to_cam
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = Rotation.from_matrix(R_cam_to_world).as_quat(
        scalar_first=True
    )  # blender wants (w, x, y, z)
    bpy.context.scene.camera = camera  # set as active camera

    return


def setup_composite_nodes():
    scene = bpy.context.scene

    tree = bpy.data.node_groups.new("My new comp", "CompositorNodeTree")
    scene.compositing_node_group = tree

    # Minimal compositor: Render Layers -> Composite (and Viewer for debugging)
    rlayers = tree.nodes.new(type="CompositorNodeRLayers")
    output = tree.nodes.new(type="NodeGroupOutput")
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    tree.links.new(rlayers.outputs["Image"], output.inputs["Image"])
    rlayers.location[0] -= 1.5 * rlayers.width

    # Add nodes
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    math_greater_than = tree.nodes.new(type="CompositorNodeMath")
    set_value = tree.nodes.new(type="CompositorNodeValue")
    map_range = tree.nodes.new(type="CompositorNodeMapRange")
    normalize = tree.nodes.new(type="CompositorNodeNormalize")
    return


def render_scene():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    # Disable all post-processing effects to get exact pixel values
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    # Enable the depth pass on the view layer
    scene.view_layers[0].use_pass_z = True

    # set background to white with no strength
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0  # strength

    # Disable anti-aliasing for exact pixel values
    scene.eevee.taa_render_samples = 1
    scene.eevee.taa_samples = 1

    # Fix lighting to get exact shading
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"

    # Save original output path and format
    orig_filepath = scene.render.filepath
    # Use multilayer EXR to capture all passes
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "OPEN_EXR"
    orig_format = scene.render.image_settings.file_format

    # Render to a temp file
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        tmp_path = f.name

    try:
        scene.render.filepath = tmp_path
        bpy.ops.render.render(animation=False, write_still=True)

        # Load the image back into memory
        img = bpy.data.images.load(tmp_path)
        width, height = img.size

        buf = np.empty(width * height * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        arr = np.flipud(buf.reshape((height, width, 4)))

        # Clean up the in-memory image datablock
        bpy.data.images.remove(img)

    finally:
        # Restore original render settings
        scene.render.filepath = orig_filepath
        scene.render.image_settings.file_format = orig_format
        os.unlink(tmp_path)

    # arr = (arr * 255).astype(np.uint8)[:, :, :3]  # convert to uint8 RGB
    # print(arr)
    return arr


def ori_vec_to_quat(ori_vecs: np.ndarray, base_vec: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """Convert orientation vectors to quaternions

    :param ori_vecs: Orientation vectors
    :type ori_vecs: np.ndarray
    :param base_vec: Base vector to align from, defaults to np.array([0, 0, 1])
    :type base_vec: np.ndarray, optional
    :return: Quaternions
    :rtype: np.ndarray
    """
    norms = np.linalg.norm(ori_vecs, axis=1, keepdims=True)
    ori_vecs = ori_vecs / norms
    angles = np.arccos(np.dot(ori_vecs, base_vec) / (np.linalg.norm(ori_vecs, axis=1) * np.linalg.norm(base_vec)))
    rot_vecs = np.cross(base_vec, ori_vecs)
    rot_vec_norms = np.linalg.norm(rot_vecs, axis=1, keepdims=True)
    rot_vec_norms = np.divide(rot_vecs, rot_vec_norms, where=(rot_vec_norms > 1e-10), out=np.zeros_like(rot_vecs))
    rot_vecs = rot_vec_norms * angles[:, np.newaxis]
    quats = Rotation.from_rotvec(rot_vecs).as_quat()
    return quats


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
    quats = ori_vec_to_quat(orientations)

    points_repeated = np.repeat(points, quats.shape[0], axis=0)
    quats_tiled = np.tile(quats, (points.shape[0], 1))
    poses = np.hstack((points_repeated, quats_tiled))

    # Return only unique poses
    poses = np.unique(poses, axis=0)
    return poses


def generate_discrete_pts(x_range, y_range, z_range, size):
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], int(size / 2))
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


def main():
    start_pos = np.array([0.008200000040233135, 0.8946679830551147, 1.6800618171691895])
    start_ori = np.array([2.23378993e-16, -1.00000000e00, 1.05291348e-31])

    x_range = (-0.2, 0.2)  # these poses are relative to the robot base
    y_range = (0.4, 0.8)
    z_range = (1.0, 1.6)
    theta_range = (-np.pi / 4, np.pi / 4)  # limit to fixed orientation for now
    phi_range = (-np.pi / 4, np.pi / 4)
    points_per_axis = 4
    angles_per_axis = 4
    discrete_poses = generate_discrete_poses(
        x_range=x_range,  # these poses are relative to the robot base
        y_range=y_range,
        z_range=z_range,
        theta_range=theta_range,
        phi_range=phi_range,
        points_per_axis=points_per_axis,
        angles_per_axis=angles_per_axis,
        start_orientation=start_ori,
    )

    bpy.context.preferences.view.show_splash = False
    clear_all_objects()
    obj_path = "/home/luke/dev/pybullet/pybullet-tree-sim/pybullet_tree_sim/meshes/trees/obj/lpy_envy_00001.obj"
    bpy_object = create_obj_from_file(filepath=obj_path)
    create_obj_with_vertex_colors(name="lpy_envy_00001", obj=bpy_object, type="emission")

    setup_composite_nodes()
    import sys

    sys.exit(0)

    camera = create_camera_with_intrinsics(
        name="camera",
        fx_mm=8.879482527283974,
        fy_mm=8.879482527283974,
        width=8,
        height=8,
        z_near=0.03,
        z_far=3.4,
    )

    # extrinsics = np.array(
    #     [[ 0.83330391, -0.23630471, -0.49976455, -0.19999295,],
    #     [ 0.23575009, -0.66580506,  0.7079022,   0.40792723,],
    #     [-0.50002632, -0.70771723, -0.4991092,   1.00349868,],
    #     [ 0.,          0.,          0.,          1.        ]]
    # )

    # extrinsics = np.identity(4)
    # extrinsics[:, 3] = [-0.05, 1, 1, 1]

    # orientation = Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()
    # extrinsics[:3, :3] = orientation

    for pose in discrete_poses:
        tf_world_to_cam = np.eye(4)
        tf_world_to_cam[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        tf_world_to_cam[:3, 3] = pose[:3]
        update_camera_position(camera=camera, extrinsics=tf_world_to_cam)

        # scene = bpy.context.scene
        # print("Scene collection objects:", [o.name for o in scene.collection.objects])
        # print("All scene objects:", [o.name for o in scene.objects])
        render_scene()
        # input("Press Enter to continue to the next pose...")
        #
        #
    return


if __name__ == "__main__":
    main()
