import logging
import numpy as np
import json
import trimesh

from pathlib import Path
from typing import Dict, Tuple

from balanna.trimesh import show_trajectory


def __parse_colors(json_dict: Dict[str, float], key: str = "color", default: Tuple[float, float, float] = (1, 0, 0)
                   ) -> Tuple[float, float, float]:
    if key not in json_dict:
        logging.debug(f"Color not found, using default color {default} using key {key}")
        return default
    if len(json_dict[key]) != 3:
        logging.warning(f"Invalid color {json_dict[key]}, using default color {default}")
        return default
    return json_dict[key]


def __parse_poses(json_dict: Dict[str, float], key: str = "poses") -> np.ndarray:
    if key not in json_dict:
        logging.warning(f"Array of positions not found in {json_dict} using key {key}")
        return None
    if len(json_dict[key]) == 0:
        logging.warning(f"Array of positions is empty in {json_dict} using key {key}")
        return None

    num_poses = len(json_dict[key])
    poses = np.zeros((num_poses, 7), dtype=np.float32)
    for k, pose in enumerate(json_dict[key]):
        if not isinstance(pose, dict):
            logging.warning(f"Invalid type of poses elements, should be dicts, got {type(pose)}")
            return None
        if "position" not in pose.keys():
            logging.warning("Invalid poses element, `position` not found")
            return None

        position = pose["position"]
        if len(position) != 3:
            logging.warning(f"Invalid poses element, got invalid `position` array {position}")
            return None

        if "orientation" in pose.keys():
            orientation = pose["orientation"]
            if len(orientation) != 4:
                logging.warning(f"Invalid poses element, got invalid `orientation` array {orientation}, "
                                f"expected quaternion")
                return None
        else:
            orientation = np.array([0, 0, 0, 1], dtype=np.float32)

        poses[k, :3] = position
        poses[k, 3:] = orientation
    return poses


def parse_scene_from_json(file_path: Path):
    with open(file_path) as f:
        data = json.load(f)

    scene = trimesh.Scene()
    for name, values in data.items():
        if "type" not in values:
            logging.warning(f"Invalid object {name}, no type found, skipping")
            continue
        object_type = values["type"]

        if object_type == "trajectory":
            positions = __parse_poses(values, "poses")[:, :3]
            color = __parse_colors(values, "color")
            scene = show_trajectory(values, scene=scene)
