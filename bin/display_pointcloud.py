import numpy as np
import pickle as pkl
import trimesh
from pathlib import Path
import argparse
from typing import List, Dict, Optional
from balanna.utils.opengl import to_opengl_transform
from balanna.display_scenes import display_scenes


VIEW_POINT = np.array([[-0.50392321, -0.18970999, -0.84265742, 11.00678718],
                       [ 0.15282067, -0.97977334,  0.12918999, -0.0317945 ],
                       [-0.8501219 , -0.06367364,  0.52272213, -8.85028203],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]])
VIEW_POINT = np.array([[-0.32293108, -0.94264604, -0.08446284,  1.42127091],
                       [-0.04442971,  0.1042454 , -0.99355871, 14.2508763 ],
                       [ 0.94537904, -0.31709832, -0.07554557,  1.13081192],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]])





def read_marker_folder(dataset_dir: Path, marker_dir: str = 'markers') -> Dict[int, np.ndarray]:
    marker_fnames = sorted([f for f in (dataset_dir.expanduser() / marker_dir).iterdir()])

    pointclouds = {}
    
    for fname in marker_fnames:
        data = np.load(fname.as_posix())

        timestamp = str(data['timestamp'])
        pointclouds[timestamp] = data['markers']
        
    return pointclouds


def trimesh_show_pointcloud(vertices: np.ndarray, view_point: Optional[np.ndarray] = None) -> trimesh.Scene:
    scene = trimesh.Scene()

    pc = trimesh.points.PointCloud(vertices.reshape(-1, 3))

    scene.add_geometry(pc, transform=np.eye(4), node_name='markers')

    scene.camera_transform = to_opengl_transform(VIEW_POINT)
    if view_point:
        scene.camera_transform = to_opengl_transform(view_point)

    return scene


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to visualize pointcloud markers in a trimesh scene.')
    parser.add_argument('-d', '--dir', type=Path, required=True, help='Root directory of dataset')

    args = parser.parse_args()
    return args


def run_all_pointclouds(opts):
    pcs = read_marker_folder(opts.dir)

    for ts, markers in pcs.items():
        print(f'current timestamp: {ts}')
        scene = trimesh_show_pointcloud(markers)

        yield {'pointcloud': scene}


if __name__ == '__main__':
    args = _parse_args()
    all_scenes = run_all_pointclouds(args)
    display_scenes(all_scenes)

