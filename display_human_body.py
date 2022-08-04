import numpy as np
import pickle as pkl
import trimesh
from pathlib import Path
import argparse
import torch
from typing import List, Dict, Optional, Any
from opengl import to_opengl_transform
from display_scenes import display_scenes
from smplx import SMPL


SMPL_MODEL_DIR = Path('~/git/smpl/models/').expanduser().as_posix()


def get_body_model(batch_size: int, dtype=torch.float32):
    smpl = SMPL(SMPL_MODEL_DIR,
                batch_size=batch_size,
                create_transl=False,
                gender='neutral',
                dtype=dtype)
    return smpl


def trimesh_show_mesh(vertices: np.ndarray, faces: np.ndarray,
        T_WB: np.ndarray=np.eye(4)) -> trimesh.Scene:
    scene = trimesh.Scene()

    human = trimesh.Trimesh(vertices, faces)
    human.visual.face_colors[:, :3] = [224, 120, 120]
    scene.add_geometry(human, transform=T_WB, node_name=f'human')
    scene.camera_transform = to_opengl_transform(VIEW_POINT)

    return scene


def read_human_body_folder(dataset_dir: Path) -> Dict[int, np.ndarray]:
    marker_fnames = sorted([f for f in dataset_dir.expanduser().glob('*.pkl')])

    human_body_parameters = {}

    import pdb;pdb.set_trace()
    for fname in marker_fnames:
        with fname.open('rb') as f:
            data = pkl.load(f, encoding='latin1')

        frame = {}
        frame['theta'] = data['fullpose']
        frame['beta'] = data['betas']
        frame['trans'] = data['trans']

        timestamp = int(fname.stem)
        human_body_parameters[timestamp] = frame
        
    return human_body_parameters


def prepare_human_bodies_for_trimesh(human_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    batch_size = len(human_data)
    smpl = get_body_model(batch_size, dtype=torch.float32)

    # collate body parameters
    thetas = torch.empty(batch_size, 72)
    betas = torch.empty(batch_size, 10)
    timestamps = torch.empty(batch_size, dtype=torch.int64)
    for i, (ts, fr) in enumerate(human_data.items()):
        thetas[i] = fr['theta']
        betas[i] = fr['beta']
        timestamps[i] = ts

    smpl_out = smpl(betas=betas, global_orient=thetas[:, :3], body_pose=thetas[:, 3:], pose2rot=True)

    return {'vertices': smpl_out.vertices.cpu().numpy(),
            'joints': smpl_out.joints.cpu().numpy(),
            'faces': smpl.faces,
            'timestamps': timestamps,
            'trans': human_data['trans']}


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to visualize a human body sequence in a trimesh scene.')
    parser.add_argument('-d', '--dir', type=Path, required=True, help='Root directory of dataset')

    args = parser.parse_args()
    return args


def run_all_human_bodies(opts):
    human_dict = read_human_body_folder(opts.dir)
    human_meshes = prepare_human_bodies_for_trimesh(human_dict)

    for i, ts in enumerate(human_meshes['timestamps']):
        print(f'current timestamp: {ts}')
        T_WB = np.eye(4)
        T_WB[:3, 3] = human_meshes['trans'][i]
        scene = trimesh_show_mesh(human_meshes['vertices'][i], human_meshes['faces'], T_WB)

        yield {'human_mesh': scene}


if __name__ == '__main__':
    args = _parse_args()
    all_scenes = run_all_human_bodies(args)
    display_scenes(all_scenes)

