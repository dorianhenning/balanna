import argparse
import pathlib
import pickle as pkl

from balanna.window_dataset import display_dataset


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path, help="cached data directory")
    parser.add_argument("--fps", type=int, help="displaying fps", default=10)
    parser.add_argument("--print-frame-index", action="store_true")
    return parser.parse_args()


def main(args):
    if not args.directory.exists():
        raise NotADirectoryError(f"Input directory {args.directory} not found")

    files = sorted(list(args.directory.glob("*.pkl")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found, invalid or empty cache directory")

    for k, file in enumerate(files):
        with open(file, 'rb') as f:
            scene_dict = pkl.load(f)
        if args.print_frame_index and "frame_index" not in scene_dict.keys():
            scene_dict["frame_index"] = str(k)
        yield scene_dict


if __name__ == '__main__':
    args_ = _parse_args()
    display_dataset(main(args_), fps=args_.fps)
