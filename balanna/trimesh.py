import os
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
    point_size: float = 4,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    """Add vertices as trimesh point cloud to the scene.

    Trimesh expects a flat point cloud of shape (N, 3). Therefore, if the point cloud is not flat, it will be
    flattened before passing it to trimesh.

    Args:
        vertices: point cloud to visualize (..., 3).
        transform: transform point cloud before adding it to scene (4, 4).
        colors: RGB color for each point (..., 3) as uint8 color 0...255..
        name: scene node name of point cloud mesh.
        point_size: point size.
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
    pc = trimesh.PointCloud(vertices_flat, colors=colors, metadata={"point_size": point_size})
    scene.add_geometry(pc, transform=transform, node_name=name)
    return scene


def show_trajectory(
    trajectory: np.ndarray,
    colors: Tuple[float, float, float] = (255, 0, 0),
    alpha: int = 255,
    fade_out: bool = False,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    """Add a colored trajectory to the scene

    Args:
        trajectory: path to be drawn (N, 3).
        colors: path color (uniformly colored line) as RGB tuple.
        alpha: path alpha value. If fade_out, max alpha value.
        fade_out: fade out line by linearly decreasing the alpha value along the path length.
        scene: scene to add path to, if None a new scene will be created.
    """
    if len(trajectory.shape) != 2 or trajectory.shape[-1] != 3:
        raise ValueError(f"Invalid size of trajectory, expected (N, 3), got {trajectory.shape}")
    if scene is None:
        scene = trimesh.Scene()

    # If the trajectory just contains a single point, then it cannot be drawn. Just return the scene.
    if trajectory.shape[0] < 2:
        print("\033[93m" + "Trajectory must contain at least two points, skipping ..." + "\033[0m")
        return scene

    # In trimesh a path consists of line segments, each connecting one vertex with the next one. The
    # way of connection is determined from the entities (similar to faces in 3D meshes).
    # Each line segment could have a different color, but vedo just supports uniformly colored lines,
    # which would be supported, but over-complicate things here.
    # process = False, otherwise vertices are weirdly reconnected
    num_points = trajectory.shape[0]
    entities = [trimesh.path.entities.Line([j, j + 1]) for j in range(num_points - 1)]

    if fade_out:
        entity_colors = [(*colors, alpha * ((k + 2) / num_points)) for k in range(num_points - 1)]
    else:
        entity_colors = [(*colors, alpha) for _ in range(num_points - 1)]
    path = trimesh.path.Path3D(entities=entities, vertices=trajectory, colors=entity_colors, process=False)

    scene.add_geometry(path)
    return scene


def show_camera(
    transform: np.ndarray,
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    """Load camera mesh and transform it to given pose.

    Args:
        transform: camera pose as transformation matrix (4, 4).
        name: scene node name of the camera mesh.
        scene: scene to add camera to, if None a new scene will be created.
    """
    if scene is None:
        scene = trimesh.Scene()
    if transform.shape != (4, 4):
        raise ValueError(f"Expected single transformation matrix, got {transform.shape}")
    script_path = os.path.basename(os.path.dirname(__file__))
    cam_mesh_path = os.path.join(script_path, "..", "meshes", "cam_z.obj")
    cam_mesh = trimesh.load_mesh(cam_mesh_path)  # already in camera orientation (z forward, x right, y down)
    cam_mesh = cam_mesh.apply_transform(transform)
    scene.add_geometry(cam_mesh, node_name=name)
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


def show_sphere(center: np.ndarray, radius: float, color: Tuple[float, float, float], scene: trimesh.Scene
                ) -> trimesh.Scene:
    """Add a sphere to the scene.

    Args:
        center: sphere center coordinates (3).
        radius: sphere radius.
        color: RGB sphere color in 0...255 (3).
        scene: scene to add sphere to.
    """
    if scene is None:
        scene = trimesh.Scene()
    if radius < 0.01:
        raise ValueError(f"Invalid sphere radius {radius}, must be > 0.01")

    sphere = trimesh.creation.uv_sphere(radius=radius)
    color_uint8 = np.array(color, dtype=np.uint8)
    sphere.visual.face_colors = np.repeat(color_uint8[None, :], len(sphere.faces), axis=0)
    sphere.apply_translation(center)

    scene.add_geometry(sphere)
    return scene


def show_capsule(p1: np.ndarray, p2: np.ndarray, radius: float, color: Tuple[float, float, float], scene: trimesh.Scene
                 ) -> trimesh.Scene:
    """Add a capsule to the scene.

    Args:
        p1: capsule start point (3).
        p2: capsule end point (3).
        radius: capsule radius.
        color: RGB capsule color in 0...255 (3).
        scene: scene to add capsule to.
    """
    if scene is None:
        scene = trimesh.Scene()
    if radius < 0.01:
        raise ValueError(f"Invalid capsule radius {radius}, must be > 0.01")
    if np.allclose(p1, p2):
        raise ValueError(f"Invalid capsule points {p1}, {p2}, must be different")

    # The canonical capsule primitive is defined as a cylinder with its z-axis aligned with the world z-axis.
    # It is determined by a single point P in the center of the cylinder and a height h.
    # This function computes the transformation from this canonical capsule to the capsule defined here from the
    # points p1 and p2. The canonical representation is given by:
    # - The origin of the capsule (P = center).
    # - The height of the capsule (h).
    # - The radius of the capsule (r).
    # - The orientation from the canonical representation (Rc).
    dp = p2 - p1
    height = np.linalg.norm(dp)
    z_axis = np.array([0, 0, 1])
    dp_normed = dp / height

    # To compute the rotation matrix that rotates the z-axis to dp, we use the Rodrigues formula.
    # See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(z_axis, dp_normed)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, dp_normed)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])  # skew-symmetric matrix of v

    Rc = np.eye(3) + vx + np.matmul(vx, vx) * (1 - c) / (s ** 2 + 1e-6)  # Rodrigues formula
    center = p1 + dp / 2  # equals P

    Tz = np.eye(4)
    Tz[:3, 3] = center
    Tz[:3, :3] = Rc

    capsule_mesh = trimesh.creation.capsule(radius=radius, height=height, transform=Tz)
    scene.add_geometry(capsule_mesh)
    return scene
