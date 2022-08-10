import numpy as np
import trimesh
import trimesh.creation

from typing import Optional


def show_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    T_WB: np.ndarray = np.eye(4),
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    human = trimesh.Trimesh(vertices, faces)
    human.visual.face_colors[:, :3] = [224, 120, 120]
    scene.add_geometry(human, transform=T_WB, node_name=name)
    return scene


def show_point_cloud(vertices: np.ndarray, scene: Optional[trimesh.Scene] = None) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    pc = trimesh.points.PointCloud(vertices.reshape(-1, 3))
    scene.add_geometry(pc, transform=np.eye(4), node_name='markers')
    return scene


def show_axis(transform: np.ndarray, scene: Optional[trimesh.Scene] = None) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    if transform.shape != (4, 4):
        raise ValueError(f"Expected single transformation matrix, got {transform.shape}")
    scene.add_geometry(trimesh.creation.axis(transform=transform))
    return scene
