import argparse
import pathlib
import pickle as pkl

from balanna.window_dataset import display_dataset


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path, help="cached data directory")
    parser.add_argument("--fps", type=int, help="displaying fps", default=10)
    parser.add_argument("--use-scene-cam", action="store_true", help="use scene camera")
    return parser.parse_args()


def main(args):
    if not args.directory.exists():
        raise NotADirectoryError(f"Input directory {args.directory} not found")

    files = sorted(list(args.directory.glob("*.pkl")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found, invalid or empty cache directory")

    scenes = []
    for k, file in enumerate(files):
        with open(file, 'rb') as f:
            scene_dict = pkl.load(f)
        scenes.append(scene_dict)
    return scenes


if __name__ == '__main__':
    args_ = _parse_args()
    display_dataset(main(args_), fps=args_.fps, use_scene_cam=args_.use_scene_cam)
