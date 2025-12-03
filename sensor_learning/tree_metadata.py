#!/usr/bin/env python3
from dataclasses import dataclass
import json
import os
import pprint as pp

from sensor_learning.trimesh_render import RenderScene

__here__ = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Cylinder:
    part_name: str
    part_id: int
    color: tuple[int, int, int]
    centroid: tuple[float, float, float]
    radius: float
    length: float
    orientation: tuple[float, float, float]


@dataclass
class Face:
    vertices: list[tuple[float, float, float]]
    normal: tuple[float, float, float]
    color: tuple[int, int, int]
    cylinder_part_name: str
    cylinder_part_id: int


@dataclass
class Limb:
    """Logical grouping of cylinders that belong to the same limb/branch.

    Holds a list of Cylinder objects and convenience methods.
    """

    name: str
    cylinders: list[Cylinder]

    def get_by_id(self, part_id: int) -> Cylinder | None:
        for c in self.cylinders:
            if c.part_id == part_id:
                return c
        return None


@dataclass
class TreeMetadata:
    def __init__(
        self,
        namespace: str,
        tree_id: int,
        tree_type: str,
        tree_metadata_path: str = os.path.join(os.path.dirname(f"{__here__}"), "data"),
    ) -> None:

        self.namespace = namespace
        self.tree_id = str(tree_id).zfill(5)
        self.tree_type = tree_type
        self.tree_metadata_path = tree_metadata_path

        metadata = self._load_tree_metadata()

        self.limbs = self.get_limbs(data=metadata)
        self.cylinders = []
        return

    def _load_tree_metadata(self):
        with open(
            os.path.join(self.tree_metadata_path, f"{self.namespace}_{self.tree_type}_{self.tree_id}_colors.json"), "r"
        ) as f:
            metadata = json.load(f)
        return metadata

    def get_limbs(self, data: dict) -> dict[str, Limb]:
        """Group cylinders by their part name.

        :param data: The cylinder data to group.
        :type data: dict
        :return: A dictionary mapping part names to Limb objects.
        :rtype: dict[str, Limb]
        """
        # Get parts dictionary
        parts_dict = {}
        for color_str, cylinder_data in data.items():
            color_tuple = tuple(map(int, color_str.strip("()").split(", ")))
            cylinder_data["part_name"] = cylinder_data["part_name"].strip().lower()
            part_name = cylinder_data["part_name"]

            if part_name not in parts_dict:
                parts_dict[part_name] = []

            cylinder_data["part_id"] = len(parts_dict[part_name])

            parts_dict[part_name].append({"color": color_tuple, **cylinder_data})

        # Get cylinder IDs, create Cylinder objects
        all_cylinders = []
        for part_name, cylinders in parts_dict.items():
            try:
                # Sort by z-coordinate (centroid[2]) - lowest first (base)
                cylinders.sort(key=lambda c: c["centroid"][2])
                for part_id, cyl_data in enumerate(cylinders):
                    cylinder = Cylinder(
                        part_name=part_name,
                        part_id=part_id,
                        color=cyl_data["color"],
                        centroid=tuple(cyl_data["centroid"]),
                        radius=cyl_data["radius"],
                        length=cyl_data["length"],
                        orientation=tuple(cyl_data["orientation"]),
                    )
                    all_cylinders.append(cylinder)
            except KeyError as e:
                pass  # Skip cylinders with missing data since they've been pruned

        # Group cylinders by part name
        limbs: dict[str, list["Cylinder"]] = {}
        for cylinder in all_cylinders:
            if cylinder.part_name not in limbs:
                limbs[cylinder.part_name] = []
            limbs[cylinder.part_name].append(cylinder)

        # Wrap into Limb dataclasses for nicer API
        limb_objs: dict[str, Limb] = {}
        for name, cyls in limbs.items():
            limb_objs[name] = Limb(name=name, cylinders=cyls)
        return limb_objs


def main():
    tree_meta = TreeMetadata(namespace="lpy", tree_id=0, tree_type="Envy")
    pp.pprint(tree_meta.limbs)
    # print(tree_meta._load_tree_metadata())
    return


if __name__ == "__main__":
    main()
