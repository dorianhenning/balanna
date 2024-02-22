import argparse
import numpy as np
import pathlib
import pickle as pkl
import tqdm
import sys

from loguru import logger

from balanna.json_parser import load_scene_from_json
from balanna.trimesh import show_point_cloud
from balanna.window_dataset import display_dataset


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pointcloud", "scenes", "json"])
    parser.add_argument("directory", type=pathlib.Path, help="Data directory or file")
    parser.add_argument("--fps", type=int, help="displaying fps", default=10)
    parser.add_argument("--use-scene-cam", action="store_true", help="use scene camera")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


def main(args):
    if not args.directory.exists():
        raise FileNotFoundError(f"Input directory {args.directory} not found")

    # Check if input is a single file or a directory of files.
    if args.directory.is_file():
        files = [args.directory]
    else:
        if args.mode == "json":
            suffix = ".json"
        elif args.mode == "pointcloud":
            suffix = ".txt"
        else:
            suffix = ".pkl"
        files = sorted(list(args.directory.glob("*" + suffix)))
        if len(files) == 0:
            raise FileNotFoundError(f"No files found, invalid or empty cache directory {args.directory}")

    # Load and process files into scenes.
    scenes = []
    if args.mode == "pointcloud":
        for k, file in enumerate(files):
            point_cloud = np.loadtxt(file).reshape(-1, 6)
            scene = show_point_cloud(point_cloud[:, :3], colors=point_cloud[:, 3:])
            scenes.append({'scene': scene})

    elif args.mode == "scenes":
        for k, file in enumerate(files):
            with open(file, 'rb') as f:
                scene_dict = pkl.load(f)
            scenes.append(scene_dict)
    elif args.mode == "json":
        for k, file in enumerate(files):
            scene_dict = load_scene_from_json(file)
            scenes.append(scene_dict)
    else:
        raise ValueError(f"Invalid displaying mode {args.mode}")

    return scenes


if __name__ == '__main__':
    args_ = _parse_args()

    # Setup logger according to debug mode.
    logger.remove()
    if args_.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Construct and display scenes.
    display_dataset(main(args_), fps=args_.fps, use_scene_cam=args_.use_scene_cam)
