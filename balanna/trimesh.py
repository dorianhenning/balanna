import numpy as np
import trimesh
import trimesh.creation

from typing import Optional


def show_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray = np.eye(4),
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    """Create a mesh from vertices and faces and add it to the scene.

    Args:
        vertices: mesh vertices (N, 3).
        faces: mesh faces.
        transform: transform mesh before adding it to scene (4, 4).
        name: scene node name of mesh.
        scene: scene to add mesh to, if None a new scene will be created.
    """
    if scene is None:
        scene = trimesh.Scene()
    human = trimesh.Trimesh(vertices, faces)
    human.visual.face_colors[:, :3] = [224, 120, 120]
    scene.add_geometry(human, transform=transform, node_name=name)
    return scene


def show_point_cloud(
    vertices: np.ndarray,
    transform: np.ndarray = np.eye(4),
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    """Add vertices as trimesh point cloud to the scene.

    Trimesh expects a flat point cloud of shape (N, 3). Therefore, if the point cloud is not flat, it will be
    flattened before passing it to trimesh.

    Args:
        vertices: point cloud to visualize (..., 3).
        transform: transform point cloud before adding it to scene (4, 4).
        name: scene node name of point cloud mesh.
        scene: scene to add point cloud mesh to, if None a new scene will be created.
    """
    if scene is None:
        scene = trimesh.Scene()
    if transform.shape != (4, 4):
        raise ValueError(f"Expected single transformation matrix, got {transform.shape}")
    pc = trimesh.points.PointCloud(vertices.reshape(-1, 3))
    scene.add_geometry(pc, transform=transform, node_name=name)
    return scene


def show_axis(
    transform: np.ndarray,
    name: Optional[str] = None,
    size: float = 0.04,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    """Add coordinate axis as trimesh mesh to the scene.

    Args:
        transform: axis pose as transformation matrix (4, 4).
        name: scene node name of axis mesh.
        size: axis origin size and radius.
        scene: scene to add axis to, if None a new scene will be created.
    """
    if scene is None:
        scene = trimesh.Scene()
    if transform.shape != (4, 4):
        raise ValueError(f"Expected single transformation matrix, got {transform.shape}")
    axis_mesh = trimesh.creation.axis(size)
    scene.add_geometry(axis_mesh, transform=transform, node_name=name)
    return scene
