#!/usr/bin/env python3
"""Hey kids! Render with Blender here!"""

import bpy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats.distributions import t
import tempfile
import os

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


def render_scene():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    # Disable all post-processing effects to get exact pixel values
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    # Enable the depth pass on the view layer
    scene.view_layers[0].use_pass_z = True
    
    # set background to white with no strength
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0  # strength
    
    # Disable anti-aliasing for exact pixel values
    scene.eevee.taa_render_samples = 1
    scene.eevee.taa_samples = 1
    
    # Fix lighting to get exact shading
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'

    # Save original output path and format
    orig_filepath = scene.render.filepath
    # Use multilayer EXR to capture all passes
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = 'OPEN_EXR'
    orig_format = scene.render.image_settings.file_format

    # Render to a temp file
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        tmp_path = f.name

    try:
        scene.render.filepath = tmp_path
        bpy.ops.render.render(write_still=True)

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
    
    arr = (arr * 255).astype(np.uint8)[:, :, :3]  # convert to uint8 RGB
    print(arr)
    return arr


def main():
    bpy.context.preferences.view.show_splash = False
    clear_all_objects()
    obj_path = "/home/luke/dev/pybullet/pybullet-tree-sim/pybullet_tree_sim/meshes/trees/obj/lpy_envy_00001.obj"
    bpy_object = create_obj_from_file(filepath=obj_path)
    create_obj_with_vertex_colors(name="lpy_envy_00001", obj=bpy_object, type="emission")

    camera = create_camera_with_intrinsics(
        name="camera",
        fx_mm=800.0,
        fy_mm=800.0,
        width=8,
        height=8,
        z_near=0.03,
        z_far=3.4,
    )
    
    extrinsics = np.array(
        [[ 0.83330391, -0.23630471, -0.49976455, -0.19999295,],
        [ 0.23575009, -0.66580506,  0.7079022,   0.40792723,],
        [-0.50002632, -0.70771723, -0.4991092,   1.00349868,],
        [ 0.,          0.,          0.,          1.        ]]
    )
    
    extrinsics = np.identity(4)
    extrinsics[:, 3] = [-0.05, 1, 1, 1]
    
    orientation = Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()
    extrinsics[:3, :3] = orientation
    update_camera_position(camera=camera, extrinsics=extrinsics)
    
    scene = bpy.context.scene
    
    print("Scene collection objects:", [o.name for o in scene.collection.objects])
    print("All scene objects:", [o.name for o in scene.objects])
    render_scene()
    return


if __name__ == "__main__":
    main()
