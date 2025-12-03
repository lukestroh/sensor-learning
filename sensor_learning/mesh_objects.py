#!/usr/bin/env python3
from dataclasses import dataclass
import trimesh
import numpy as np


class MeshObjects:
    @staticmethod
    def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
        """Load a mesh from a path

        :param mesh_path: Mesh path
        :type mesh_path: str
        :return: Trimesh object of the mesh
        :rtype: trimesh.Trimesh
        """
        mesh = trimesh.load(mesh_path, process=False)  # process=False keeps vertex sharing as in file
        return mesh

    @staticmethod
    def color_unique_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Duplicate vertices per-face so each face can get a flat color (no vertex-sharing)
        This ensures a unique set of vertices per triangle so color doesn't interpolate across adjacent faces.

        :param mesh: Trimesh object
        :type mesh: trimesh.Trimesh
        :return: A recolored trimesh object
        :rtype: trimesh.Trimesh
        """
        faces = mesh.faces
        n_faces = faces.shape[0]
        verts_per_face = mesh.vertices[faces]  # shape (n_faces, 3, 3)
        new_vertices = verts_per_face.reshape(-1, 3)  # (n_faces*3, 3)
        new_faces = np.arange(len(new_vertices)).reshape(-1, 3)  # (n_faces, 3)

        # Assign random color per face
        face_colors = np.random.randint(0, 256, size=(n_faces, 4), dtype=np.uint8)
        face_colors[:, 3] = 255  # Set alpha to 255

        # Expand face colors to match new vertices
        vertex_colors = np.repeat(face_colors, 3, axis=0)  # (n_faces*3, 4)

        recolored_mesh = trimesh.Trimesh(
            vertices=new_vertices, faces=new_faces, vertex_colors=vertex_colors, process=False
        )
        return recolored_mesh
