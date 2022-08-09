import numpy as np
import argparse

from pathlib import Path
from typing import Dict

from balanna.trimesh import show_point_cloud
from balanna.display_scenes import display_scenes
from balanna.utils.opengl import to_opengl_transform


VIEW_POINT = np.array([[-0.32293108, -0.94264604, -0.08446284,  1.42127091],
                       [-0.04442971,  0.1042454 , -0.99355871, 14.2508763 ],
                       [ 0.94537904, -0.31709832, -0.07554557,  1.13081192],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]])


def read_marker_folder(dataset_dir: Path, marker_dir: str = 'markers') -> Dict[int, np.ndarray]:
    marker_fnames = sorted([f for f in (dataset_dir.expanduser() / marker_dir).iterdir()])
    point_clouds = {}
    for fname in marker_fnames:
        data = np.load(fname.as_posix())
        timestamp = int(data['timestamp'])
        point_clouds[timestamp] = data['markers']
    return point_clouds


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to visualize point cloud markers in a trimesh scene.')
    parser.add_argument('directory', type=Path, help='Root directory of dataset')
    return parser.parse_args()


def run_all_point_clouds(directory: Path):
    pcs = read_marker_folder(directory)
    for ts, markers in pcs.items():
        print(f'current timestamp: {ts}')
        scene = show_point_cloud(markers)
        scene.camera_transform = to_opengl_transform(VIEW_POINT)
        yield {'point_cloud': scene}


if __name__ == '__main__':
    args = _parse_args()
    all_scenes = run_all_point_clouds(args.directory)
    display_scenes(all_scenes)
