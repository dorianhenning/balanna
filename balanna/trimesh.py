import numpy as np
import trimesh
import trimesh.creation

from typing import List, Optional, Tuple, Union


def show_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray = np.eye(4),
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None,
    face_color: Optional[np.ndarray] = None,
    vertex_color: Optional[np.ndarray] = None
) -> trimesh.Scene:
    """Create a mesh from vertices and faces and add it to the scene.

    Args:
        vertices: mesh vertices (N, 3).
        faces: mesh faces.
        transform: transform mesh before adding it to scene (4, 4).
        name: scene node name of mesh.
        scene: scene to add mesh to, if None a new scene will be created.
        face_color: face colors (F, 3).
        vertex_color: vertex colors (N, 3/4).
    """
    if vertex_color is not None and vertex_color.shape[:-1] != vertices.shape[:-1]:
        raise ValueError(f"Invalid vertex colors, must be {vertices.shape}, got {vertex_color.shape}")
    if vertex_color is not None and face_color is not None:
        raise ValueError(f"Cannot set both vertex and face color")

    if scene is None:
        scene = trimesh.Scene()

    mesh = trimesh.Trimesh(vertices, faces)
    if face_color is not None:
        mesh.visual.face_colors[:, :3] = face_color
    elif vertex_color is not None:
        mesh.visual.vertex_colors = vertex_color
    else:
        mesh.visual.face_colors[:, :3] = [224, 120, 120]

    mesh = mesh.apply_transform(transform)
    scene.add_geometry(mesh, node_name=name)
    return scene


def show_grid(
    xy_min: float = -10.0,
    xy_max: float = 10.0,
    z: float = 0.0,
    resolution: float = 1.0,
    alpha: float = 255,
    dark_color: Tuple[float, float, float] = (120, 120, 120),
    light_color: Tuple[float, float, float] = (255, 255, 255),
    transform: np.ndarray = np.eye(4),
    scene: trimesh.Scene = None
) -> trimesh.Scene:
    """Create a 2D squared grid in the xy-plane, color it in a chessboard pattern and transform it accordingly.

    Args:
        xy_min: minimal coordinate in xy direction, i.e. (xy_min, xy_min).
        xy_max: maximal coordinate in xy direction, i.e. (xy_max, xy_max).
        z: plane height at creation.
        resolution: cell size.
        alpha: color alpha value for all cells.
        dark_color: chessboard dark color.
        light_color: chessboard light color.
        transform: mesh transform after creation (in relation to center point (0, 0, z)).
        scene: scene to add mesh to, if None a new scene will be created.
    """
    if scene is None:
        scene = trimesh.Scene()

    num_points_per_row = int((xy_max - xy_min) // resolution) + 1
    w = num_points_per_row - 1
    num_faces = w ** 2

    # Compute vertices as 2D mesh-grid in the xy-plane. Set all z components to the given z value.
    vertices_x, vertices_y = np.meshgrid(np.linspace(xy_min, xy_max, num_points_per_row),
                                         np.linspace(xy_min, xy_max, num_points_per_row))
    vertices_x = vertices_x.flatten()
    vertices_y = vertices_y.flatten()
    vertices = np.stack([vertices_x, vertices_y, np.ones_like(vertices_x) * z], axis=-1)

    # Compute grid faces and face colors to create chessboard pattern.
    faces = []
    face_colors = np.zeros((num_faces, 4), dtype=np.uint8)
    face_colors[:, -1] = alpha  # alpha value
    sym = [light_color, dark_color]
    face_idx = 0
    for i in range(w):
        for j in range(w):
            # add face in counter-clockwise vertex order.
            face_ij = (i * num_points_per_row + j,
                       i * num_points_per_row + j + 1,
                       (i + 1) * num_points_per_row + j + 1,
                       (i + 1) * num_points_per_row + j)
            faces.append(face_ij)

            # if even number of elements in row: at beginning of row, swap colors.
            # if odd: order is automatically swapped after finishing one row.
            if face_idx % w == 0 and w % 2 == 0:
                sym.reverse()
            idx = 0 if face_idx % 2 == 0 else 1  # alternate index for chessboard pattern
            face_colors[face_idx, :3] = sym[idx]

            # Increment face index.
            face_idx += 1
    face_colors = np.concatenate([face_colors, face_colors])

    grid = trimesh.Trimesh(vertices, faces, face_colors=face_colors)
    scene.add_geometry(grid, transform=transform)
    return scene


def show_point_cloud(
    vertices: np.ndarray,
    transform: np.ndarray = np.eye(4),
    colors: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    """Add vertices as trimesh point cloud to the scene.

    Trimesh expects a flat point cloud of shape (N, 3). Therefore, if the point cloud is not flat, it will be
    flattened before passing it to trimesh.

    Args:
        vertices: point cloud to visualize (..., 3).
        transform: transform point cloud before adding it to scene (4, 4).
        colors: RGB color for each point (..., 3) as uint8 color 0...255..
        radius: point radius in pixels.
        name: scene node name of point cloud mesh.
        scene: scene to add point cloud mesh to, if None a new scene will be created.
    """
    if scene is None:
        scene = trimesh.Scene()
    if transform.shape != (4, 4):
        raise ValueError(f"Expected single transformation matrix, got {transform.shape}")

    vertices_flat = vertices.reshape(-1, 3)
    if colors is None:
        colors = np.zeros((len(vertices_flat), 3))
        colors[:, 0] = 255  # default = red
    pc = trimesh.PointCloud(vertices_flat, colors=colors)
    scene.add_geometry(pc, transform=transform, node_name=name)
    return scene


def show_axis(
    transform: np.ndarray,
    name: Optional[str] = None,
    size: float = 0.06,
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
    axis_mesh = trimesh.creation.axis(size, transform=transform, axis_radius=size/3)
    scene.add_geometry(axis_mesh, node_name=name)
    return scene


def show_axes(
    transforms: Union[List[np.ndarray], np.ndarray],
    sizes: Optional[List[float]] = None,
    scene: Optional[trimesh.Scene] = None
):
    """Add coordinate axes for multiple frames to the scene.

    Args:
        transforms: axes poses as transformation matrices (N, 4, 4 or list of (4, 4)).
        sizes: axes sizes.
        scene: scene to add axis to, if None a new scene will be created.
    """
    if sizes is None:
        sizes = [0.04] * len(transforms)
    for tf, size in zip(transforms, sizes):
        scene = show_axis(tf, size=size, scene=scene)
    return scene
