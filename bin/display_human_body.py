import numpy as np
import pickle as pkl
import argparse
import torch

from pathlib import Path
from typing import Dict, Any
from balanna.trimesh import show_mesh
from balanna.display_scenes import display_scenes
from smplx import SMPL


SMPL_MODEL_DIR = Path('~/git/smpl/models/').expanduser().as_posix()

VIEW_POINT = np.array([[-0.50392321, -0.18970999, -0.84265742, 11.00678718],
                       [ 0.15282067, -0.97977334,  0.12918999, -0.0317945 ],
                       [-0.8501219 , -0.06367364,  0.52272213, -8.85028203],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]])


def get_body_model(batch_size: int, dtype=torch.float32):
    smpl = SMPL(SMPL_MODEL_DIR,
                batch_size=batch_size,
                create_transl=False,
                gender='neutral',
                dtype=dtype)
    return smpl



def read_human_body_folder(dataset_dir: Path) -> Dict[int, np.ndarray]:
    marker_fnames = sorted([f for f in dataset_dir.expanduser().glob('*.pkl')])

    human_body_parameters = {}

    for fname in marker_fnames:
        if not fname.stem.endswith('_stageii'):
            continue
        with fname.open('rb') as f:
            data = pkl.load(f, encoding='latin1')

        frame = {}
        frame['theta'] = data['fullpose']
        frame['beta'] = data['betas']
        frame['trans'] = data['trans']

        timestamp = int(fname.stem.split('_')[0])
        human_body_parameters[timestamp] = frame
        
    return human_body_parameters


def prepare_human_bodies_for_trimesh(human_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    batch_size = len(human_data)
    smpl = get_body_model(batch_size, dtype=torch.float32)

    # collate body parameters
    thetas = torch.empty(batch_size, 72)
    betas = torch.empty(batch_size, 10)
    timestamps = np.empty(batch_size, dtype=np.int64)
    trans = np.empty((batch_size, 3))
    for i, (ts, fr) in enumerate(human_data.items()):
        thetas[i] = torch.from_numpy(fr['theta'])
        betas[i] = torch.from_numpy(fr['beta'])
        timestamps[i] = ts
        trans[i] = fr['trans']

    smpl_out = smpl(betas=betas, global_orient=thetas[:, :3], body_pose=thetas[:, 3:], pose2rot=True)

    return {'vertices': smpl_out.vertices.cpu().numpy(),
            'joints': smpl_out.joints.cpu().numpy(),
            'faces': smpl.faces,
            'timestamps': timestamps,
            'trans': trans}


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
        scene = show_mesh(human_meshes['vertices'][i], human_meshes['faces'], T_WB)
        yield {'human_mesh': scene}


if __name__ == '__main__':
    args = _parse_args()
    all_scenes = run_all_human_bodies(args)
    display_scenes(all_scenes)

