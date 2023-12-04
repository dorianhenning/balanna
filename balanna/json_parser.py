import numpy as np
import json
import trimesh

from loguru import logger
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional, Tuple

from balanna.trimesh import show_trajectory, show_axis, show_capsule, show_sphere


def __parse_colors(json_dict: Dict[str, List], key: str = "color", default: Tuple[float, float, float] = (1, 0, 0)
                   ) -> Tuple[float, float, float]:
    if key not in json_dict:
        logger.debug(f"Color not found, using default color {default} using key {key}")
        return default
    if len(json_dict[key]) != 3:
        logger.debug(f"Invalid color {json_dict[key]}, using default color {default}")
        return default
    red = int(json_dict[key][0] * 255)
    green = int(json_dict[key][1] * 255)
    blue = int(json_dict[key][2] * 255)
    return red, green, blue


def __parse_position(json_dict: Dict[str, List], key: str = "position") -> Optional[np.ndarray]:
    if key not in json_dict.keys():
        logger.debug(f"Invalid position element, `{key}` not found")
        return None
    position = np.array(json_dict[key])
    if len(position) != 3:
        logger.debug(f"Invalid position element, got invalid `{key}` array {position}")
        return None
    return position


def __parse_pose(json_dict: Dict[str, List]) -> Optional[np.ndarray]:
    position = __parse_position(json_dict)
    if position is None:
        return None

    if "orientation" in json_dict.keys():
        orientation = np.array(json_dict["orientation"])
        if len(orientation) != 4:
            logger.debug(f"Invalid poses element, got invalid `orientation` array {orientation}, "
                         f"expected quaternion")
            return None
    elif "orientationRPY" in json_dict.keys():
        orientation_rpy = np.array(json_dict["orientationRPY"])
        if len(orientation_rpy) != 3:
            logger.debug(f"Invalid poses element, got invalid `orientationRPY` array {orientation_rpy}, "
                         f"expected roll-pitch-yaw")
            return None
        orientation = Rotation.from_euler("XYZ", orientation_rpy, degrees=False).as_quat()
    else:
        orientation = np.array([0, 0, 0, 1], dtype=np.float32)

    return np.concatenate((position, orientation))


def __parse_poses(json_dict: Dict[str, List], key: str = "poses") -> Optional[np.ndarray]:
    if key not in json_dict:
        logger.debug(f"Array of positions not found in {json_dict} using key {key}")
        return None
    if len(json_dict[key]) == 0:
        logger.debug(f"Array of positions is empty in {json_dict} using key {key}")
        return None

    num_poses = len(json_dict[key])
    poses = np.zeros((num_poses, 7), dtype=np.float32)
    for k, pose in enumerate(json_dict[key]):
        if not isinstance(pose, dict):
            logger.debug(f"Invalid type of poses elements, should be dicts, got {type(pose)}")
            return None
        poses[k] = __parse_pose(pose)
    return poses


def load_scene_from_json(file_path: Path):
    with open(file_path) as f:
        data = json.load(f)

    scene = trimesh.Scene()
    for name, values in data.items():
        if "type" not in values:
            logger.warning(f"Invalid object {name}, no type found, skipping")
            continue
        object_type = values["type"]

        if object_type == "trajectory":
            positions = __parse_poses(values, "poses")[:, :3]
            if positions is None:
                continue
            color = __parse_colors(values, "color")
            scene = show_trajectory(positions, colors=color, scene=scene)

        elif object_type == "frame":
            pose = __parse_pose(values)
            if pose is None:
                continue
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            transform[:3, 3] = pose[:3]
            scene = show_axis(transform, name=name, scene=scene)

        elif object_type == "capsule":
            radius = values.get("radius", None)
            if radius is None:
                logger.warning(f"Invalid capsule object {name}, no radius found, skipping")
                continue
            p1 = __parse_position(values, "p1")
            p2 = __parse_position(values, "p2")
            if p1 is None or p2 is None:
                logger.warning(f"Invalid capsule object {name}, no p1/p2 found, skipping")
                continue
            color = __parse_colors(values, "color")
            scene = show_capsule(p1, p2, radius, color=color, scene=scene)

        elif object_type == "sphere":
            radius = 0.1  # values.get("radius", None)
            if radius is None:
                logger.warning(f"Invalid sphere object {name}, no radius found, skipping")
                continue
            center = __parse_position(values, "center")
            if center is None:
                logger.warning(f"Invalid sphere object {name}, no center found, skipping")
                continue
            color = __parse_colors(values, "color")
            scene = show_sphere(center, radius, color=color, scene=scene)

        else:
            logger.warning(f"Invalid object {name}, unknown type {object_type}, skipping")
            continue

    return scene
