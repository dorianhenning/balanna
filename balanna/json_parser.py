import numpy as np
import json
import traceback
import trimesh

from loguru import logger
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional, Tuple, Union

from balanna.trimesh import (
	show_trajectory, 
    show_axis, 
    show_capsule, 
    show_cylinder,
    show_ellipsoid, 
    show_plane,
    show_sphere, 
    show_point_cloud, 
    RGBorRGBAType
)


def __parse_colors(json_dict: Dict[str, List], key: str = "color", default: Tuple[int, int, int] = (255, 0, 0)
                   ) -> RGBorRGBAType:
    if key not in json_dict:
        logger.debug(f"Color not found, using default color {default} using key {key}")
        return default
    if len(json_dict[key]) not in [3, 4]:
        logger.debug(f"Invalid color {json_dict[key]}, using default color {default}")
        return default

    red = int(json_dict[key][0] * 255)
    green = int(json_dict[key][1] * 255)
    blue = int(json_dict[key][2] * 255)

    if len(json_dict[key]) == 4:
        alpha = int(json_dict[key][3] * 255)
        return red, green, blue, alpha
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


def __parse_transform(json_dict: Dict[str, List]) -> Optional[np.ndarray]:
    pose = __parse_pose(json_dict)
    if pose is None:
        return None

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


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


def __parse_radii(json_dict: Dict[str, List], key: str = "radii") -> Optional[np.ndarray]:
    if key not in json_dict:
        logger.debug(f"Radii not found in {json_dict} using key {key}")
        return None
    radii = np.array(json_dict[key])
    if len(radii) != 3:
        logger.debug(f"Invalid radii array {radii} in {json_dict} using key {key}")
        return None
    return radii


def __parse_lines_count(json_dict: Dict[str, List]) -> Optional[Tuple[int, int]]:
    latitude = json_dict.get("countLatitude", None)
    longitude = json_dict.get("countLongitude", None)
    if latitude is None or longitude is None:
        logger.debug(f"Count not found, returning None")
        return None
    if not isinstance(latitude, int) or not isinstance(longitude, int):
        logger.debug(f"Invalid count {latitude, longitude}, must be integers, returning None")
        return None
    if latitude < 3 or longitude < 3:
        logger.debug(f"Invalid count {latitude, longitude}, must be >= 3, returning None")
        return None
    return latitude, longitude


def load_scene_from_json(file_path: Path):
    with open(file_path) as f:
        data = json.load(f)

    scene_dict = dict()
    scene = trimesh.Scene()
    for name, values in data.items():
        if "type" not in values:
            logger.warning(f"Invalid object {name}, no type found, skipping")
            continue
        object_type = values["type"]
        logger.debug(f"Processing object {name} with type {object_type}")

        try:
            if object_type == "trajectory":
                positions = __parse_poses(values, "poses")[:, :3]
                if positions is None:
                    continue
                color = __parse_colors(values, "color")
                scene = show_trajectory(positions, colors=color, scene=scene)

            elif object_type == "frame":
                transform = __parse_transform(values)
                if transform is None:
                    continue
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
                count = __parse_lines_count(values)
                scene = show_capsule(p1, p2, radius, color=color, scene=scene, count=count)

            elif object_type == "sphere":
                radius = values.get("radius", None)
                if radius is None:
                    logger.warning(f"Invalid sphere object {name}, no radius found, skipping")
                    continue            
                center = __parse_position(values, "center")
                if center is None:
                    logger.warning(f"Invalid sphere object {name}, no center found, skipping")
                    continue
                count = __parse_lines_count(values)
                color = __parse_colors(values, "color")
                scene = show_sphere(center, radius, color=color, scene=scene, count=count)

            elif object_type == "ellipsoid":
                radii = __parse_radii(values, "radii")
                if radii is None:
                    logger.warning(f"Invalid ellipsoid object {name}, no radii found, skipping")
                    continue
                center = __parse_position(values, "center")
                if center is None:
                    logger.warning(f"Invalid ellipsoid object {name}, no center found, skipping")
                    continue
                color = __parse_colors(values, "color")
                count = __parse_lines_count(values)
                scene = show_ellipsoid(center, radii, color=color, scene=scene, count=count)

            elif object_type == "cylinder":
                radius = values.get("radius", None)
                if radius is None:
                    logger.warning(f"Invalid cylinder object {name}, no radius found, skipping")
                    continue
                height = values.get("height", None)
                if height is None:
                    logger.warning(f"Invalid cylinder object {name}, no height found, skipping")
                    continue
                transform = __parse_transform(values)
                if transform is None:
                    logger.warning(f"Invalid cylinder object {name}, no transform found, using identity")
                    transform = np.eye(4)
                color = __parse_colors(values, "color")
                count = values.get("count", None)
                scene = show_cylinder(transform, radius, height, color=color, scene=scene, count=count)

            elif object_type == "point_cloud":
                positions = np.array(values.get("points", []))
                if len(positions) == 0:
                    logger.warning(f"Invalid point cloud object {name}, no points found, skipping")
                    continue
                colors = np.array(values["colors"]) * 255 if "colors" in values else None
                point_size = values.get("point_size", 4.0)
                if point_size < 0.1:
                    logger.debug(f"Point size of point cloud too small, multiplying by 500")
                    point_size *= 500
                scene = show_point_cloud(positions, colors=colors, point_size=point_size, scene=scene)

            elif object_type == "message":
                title = values.get("title", "info")
                text = values.get("text", None)
                if text is None:
                    logger.warning(f"Invalid text object {name}, no text found, skipping")
                    continue
                scene_dict[title] = text

            elif object_type == "plane": 
                if "normal" not in values:
                    logger.warning(f"Invalid plane object {name}, no normal found, skipping")
                    continue
                if "center" not in values:
                    logger.warning(f"Invalid plane object {name}, no center found, skipping")
                    continue
                normal = np.array(values["normal"])
                center = np.array(values["center"])
                extent_xy = values.get("extent_xy", 5.0)
                extent_z = values.get("extent_z", 0.01)
                color = __parse_colors(values, "color")
                scene = show_plane(normal, center, extent_xy, extent_z, color=color, scene=scene)

            else:
                logger.warning(f"Invalid object {name}, unknown type {object_type}, skipping")
                continue

        except Exception as e:
            logger.error(f"Error processing object {name}, continuing")
            logger.debug(traceback.format_exc())
            continue

    if "scene" in scene_dict:
        logger.warning(f"Scene dict already has a `scene` key, overwriting")
    scene_dict["scene"] = scene
    return scene_dict
