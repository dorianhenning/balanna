import argparse
import numpy as np
import pickle as pkl
import torch

from pathlib import Path
from typing import Dict
from smplx import SMPL

from balanna.trimesh import show_mesh
from balanna.display_scenes import display_scenes


def get_body_model(path: Path, batch_size: int, dtype=torch.float32):
    return SMPL(path.as_posix(), batch_size=batch_size, create_transl=False, gender="male", dtype=dtype)


def read_human_body_folder(fname: Path, smpl_model_path: Path) -> Dict[str, np.ndarray]:
    with fname.open("rb") as f:
        data = pkl.load(f, encoding="latin1")
    thetas = torch.tensor(data["fullpose"], dtype=torch.float32)
    betas = torch.tensor(data["betas"], dtype=torch.float32)
    trans = torch.tensor(data["trans"], dtype=torch.float32)

    smpl = get_body_model(smpl_model_path, batch_size=len(thetas), dtype=torch.float32)
    smpl_out = smpl(betas=betas[None], global_orient=thetas[:, :3], body_pose=thetas[:, 3:], pose2rot=True)

    return {
        "vertices": smpl_out.vertices.cpu().numpy(),
        "joints": smpl_out.joints.cpu().numpy(),
        "faces": smpl.faces,
        "trans": trans,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to visualize a human body sequence in a trimesh scene.")
    parser.add_argument("fname", type=Path, help="Root file name to visualize")
    parser.add_argument("--smpl-model", type=Path, required=True, help="Path to smpl model")
    return parser.parse_args()


def run_all_human_bodies(fname: Path, smpl_model_path: Path):
    human_meshes = read_human_body_folder(fname, smpl_model_path)
    for i, vertices in enumerate(human_meshes['vertices']):
        T_WB = np.eye(4)
        T_WB[:3, 3] = human_meshes["trans"][i]
        scene = show_mesh(vertices, human_meshes["faces"], transform=T_WB)
        yield {"human_mesh": scene}


if __name__ == "__main__":
    args = _parse_args()
    all_scenes = run_all_human_bodies(args.fname, smpl_model_path=args.smpl_model)
    display_scenes(all_scenes)
