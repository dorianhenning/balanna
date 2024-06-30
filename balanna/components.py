import os
import numpy as np
import trimesh
import trimesh.creation

from loguru import logger
from scipy.spatial.transform import Rotation
from typing import List, Optional, Tuple, Union


__all__ = [
    "show_box_w_transform",
    "show_box_w_aabb",
    "show_ellipsoid",
    "show_plane",
    "show_mesh",
    "show_trimesh",
    "show_grid",
    "show_point_cloud",
    "show_trajectory",
    "show_camera",
    "show_axis",
    "show_axes",
    "show_sphere",
    "show_capsule",
    "show_cylinder",
]


RGBType = Tuple[int, int, int]
RGBAType = Tuple[int, int, int, int]
RGBorRGBAType = Union[RGBAType, RGBType]


def show_box_w_transform(
    extents: Tuple[float, float, float],
    transform: np.ndarray,
    color: RGBorRGBAType = (120, 120, 120),
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None,
):
    """Create a 3D box with the given transformation matrix to the plane coordinate frame (normal = z axis).

    Args:
        extents: box extents (x, y, z).
        normal: plane normal vector (3).
        point: point on the plane (3).
        color: box color as RGB / RGBA tuple in 0...255 (3/4).
        name: scene node name of mesh.
        scene: scene to add mesh to, if None a new scene will be created.
    """
    if len(extents) != 3:
        raise ValueError(f"Invalid extents, expected (3,), got {len(extents)}")
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transformation matrix of shape (4, 4), got {transform.shape}")

    if scene is None:
        scene = trimesh.Scene()
    color_uint8 = np.array(color, dtype=np.uint8)

    box = trimesh.creation.box(extents=extents, transform=transform, name=name)
    box.visual.face_colors = np.repeat(color_uint8[None, :], len(box.faces), axis=0)

    scene.add_geometry(box)
    return scene


def show_plane(
    normal: np.ndarray,
    point: np.ndarray,
    extent_xy: float = 5.0,
    extent_z: float = 0.01,
    color: RGBorRGBAType = (120, 120, 120),
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None,
):
    """Create a 3D plane as a flat 3D box with the normal vector and a point on the plane.

    Args:
        normal: plane normal vector (3).
        point: point on the plane (3).
        extent_xy: plane extent in x and y direction.
        extent_z: plane extent in z direction (thickness of the plane box).
        name: scene node name of mesh.
        scene: scene to add mesh to, if None a new scene will be created.
    """
    if normal.shape != (3,):
        raise ValueError(f"Invalid normal shape, expected (3,), got {normal.shape}")
    if point.shape != (3,):
        raise ValueError(f"Invalid point shape, expected (3,), got {point.shape}")
    unit_axis = np.array([0, 0, 1])  # normal of the unit plane in trimesh

    # Construct transform from normal vector and point on plane. The plane normal is aligned with the z-axis.
    # Therefore, the rotation matrix R aligns the unit z-axis with the plane normal.
    R, _ = Rotation.align_vectors(normal[None], unit_axis[None])
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T[:3, 3] = point

    extents = (extent_xy, extent_xy, extent_z)
    return show_box_w_transform(extents, transform=T, name=name, color=color, scene=scene)


def show_box_w_aabb(
    aabb: np.ndarray, name: Optional[str] = None, scene: Optional[trimesh.Scene] = None
):
    """Create a 3D box with the min and max corner.

    Args:
        aabb: [(x_min, y_min, z_min), (x_max, y_max, z_max)] (2, 3).
        name: scene node name of mesh.
        scene: scene to add mesh to, if None a new scene will be created.
    """
    if aabb.shape != (2, 3):
        raise ValueError(f"Invalid aabb shape, expected (2, 3), got {aabb.shape}")

    if scene is None:
        scene = trimesh.Scene()

    box = trimesh.creation.box(bounds=aabb, name=name)
    scene.add_geometry(box)
    return scene


def show_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray = np.eye(4),
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None,
    face_color: Optional[np.ndarray] = None,
    vertex_color: Optional[np.ndarray] = None,
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
    if vertex_color is not None and vertex_color.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"Invalid vertex colors, must be {vertices.shape}, got {vertex_color.shape}"
        )
    if vertex_color is not None and face_color is not None:
        raise ValueError("Cannot set both vertex and face color")

    mesh = trimesh.Trimesh(vertices, faces)
    if face_color is not None:
        mesh.visual.face_colors[:, :3] = face_color
    elif vertex_color is not None:
        mesh.visual.vertex_colors = vertex_color
    else:
        mesh.visual.face_colors[:, :3] = [224, 120, 120]

    return show_trimesh(mesh, name, transform, scene, copy_mesh=False)


def show_trimesh(
    mesh: trimesh.Trimesh,
    name: Optional[str] = None,
    transform: np.ndarray = np.eye(4),
    scene: Optional[trimesh.Scene] = None,
    copy_mesh: bool = True,
):
    """Add a trimesh mesh to the scene.

    Args:
        mesh: trimesh mesh to add.
        name: scene node name of mesh.
        transform: transform mesh before adding it to scene (4, 4).
        scene: scene to add mesh to, if None a new scene will be created.
        copy_mesh: copy the mesh before adding it to the scene and applying the transform.
    """
    if scene is None:
        scene = trimesh.Scene()
    mesh_scene = mesh.copy() if copy_mesh else mesh
    mesh_scene = mesh_scene.apply_transform(transform)
    scene.add_geometry(mesh_scene, node_name=name)
    return scene


def show_grid(
    xy_min: float = -10.0,
    xy_max: float = 10.0,
    z: float = 0.0,
    resolution: float = 1.0,
    alpha: float = 255,
    dark_color: RGBAType = (120, 120, 120),
    light_color: RGBAType = (255, 255, 255),
    transform: np.ndarray = np.eye(4),
    scene: trimesh.Scene = None,
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
    num_faces = w**2

    # Compute vertices as 2D mesh-grid in the xy-plane. Set all z components to the given z value.
    vertices_x, vertices_y = np.meshgrid(
        np.linspace(xy_min, xy_max, num_points_per_row),
        np.linspace(xy_min, xy_max, num_points_per_row),
    )
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
            face_ij = (
                i * num_points_per_row + j,
                i * num_points_per_row + j + 1,
                (i + 1) * num_points_per_row + j + 1,
                (i + 1) * num_points_per_row + j,
            )
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
    scene: Optional[trimesh.Scene] = None,
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
    colors: Union[RGBorRGBAType, np.ndarray] = (255, 0, 0),
    fade_out: bool = False,
    scene: Optional[trimesh.Scene] = None,
) -> trimesh.Scene:
    """Add a colored trajectory to the scene

    Args:
        trajectory: path to be drawn (N, 3).
        colors: path color (tuple for uniformly colored line as RGB(-A) tuple & array for segment-wise color).
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
        logger.warning("Trajectory must contain at least two points, skipping ...")
        return scene

    # In trimesh a path consists of line segments, each connecting one vertex with the next one. The
    # way of connection is determined from the entities (similar to faces in 3D meshes).
    # Each line segment could have a different color, but vedo just supports uniformly colored lines,
    # which would be supported, but over-complicate things here.
    # process = False, otherwise vertices are weirdly reconnected
    num_points = trajectory.shape[0]
    entities = [trimesh.path.entities.Line([j, j + 1]) for j in range(num_points - 1)]

    if isinstance(colors, tuple):
        alpha = colors[-1] if len(colors) == 4 else 255
        colors_rgb = colors[:3]

        alphas = [
            alpha * ((k + 2) / num_points) if fade_out else alpha for k in range(num_points - 1)
        ]
        entity_colors = [(*colors_rgb, alphas[k]) for k in range(num_points - 1)]

    elif isinstance(colors, np.ndarray):
        if colors.dtype != np.uint8:
            raise ValueError(f"Invalid colors dtype, expected uint8, got {colors.dtype}")
        if len(colors) != num_points - 1:
            raise ValueError(f"Invalid colors length, expected {num_points - 1}, got {len(colors)}")
        if len(colors.shape) != 2:
            raise ValueError(f"Invalid colors shape, expected (N-1, 3/4), got {colors.shape}")

        if colors.shape[1] == 3:
            colors_rgb = colors
            alphas = [
                255 * (k + 2) / num_points if fade_out else 255 for k in range(num_points - 1)
            ]
        elif colors.shape[1] == 4:
            if fade_out:
                raise ValueError("Fade out not supported for RGBA colors")
            colors_rgb = colors[:, :3]
            alphas = colors[:, 3]
        else:
            raise ValueError(f"Invalid colors shape, expected (N-1, 3/4), got {colors.shape}")

        entity_colors = [(*colors_rgb[k], alphas[k]) for k in range(num_points - 1)]

    # The vertices must be copied, otherwise the original trajectory might be overwritten afterwards.
    path = trimesh.path.Path3D(
        entities=entities,
        vertices=trajectory.copy(),
        colors=entity_colors,
        process=False,
    )
    scene.add_geometry(path)
    return scene


def show_camera(
    transform: np.ndarray,
    name: Optional[str] = None,
    scene: Optional[trimesh.Scene] = None,
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
    cam_mesh = trimesh.load_mesh(
        cam_mesh_path
    )  # already in camera orientation (z forward, x right, y down)
    cam_mesh = cam_mesh.apply_transform(transform)
    scene.add_geometry(cam_mesh, node_name=name)
    return scene


def show_axis(
    transform: np.ndarray,
    name: Optional[str] = None,
    size: float = 0.06,
    scene: Optional[trimesh.Scene] = None,
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
    axis_mesh = trimesh.creation.axis(size, transform=transform, axis_radius=size / 3)
    scene.add_geometry(axis_mesh, node_name=name)
    return scene


def show_axes(
    transforms: Union[List[np.ndarray], np.ndarray],
    sizes: Optional[List[float]] = None,
    scene: Optional[trimesh.Scene] = None,
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


def show_sphere(
    center: np.ndarray,
    radius: float,
    color: RGBorRGBAType,
    scene: Optional[trimesh.Scene] = None,
    count: Optional[Tuple[int, int]] = None,
) -> trimesh.Scene:
    """Add a sphere to the scene.

    Args:
        center: sphere center coordinates (3).
        radius: sphere radius.
        color: RGB / RGBA sphere color in 0...255 (3/4).
        scene: scene to add sphere to.
        count: sphere resolution as number of latitude and longitude lines (2).
    """
    if scene is None:
        scene = trimesh.Scene()
    if radius < 0.01:
        raise ValueError(f"Invalid sphere radius {radius}, must be > 0.01")
    if count is not None and (count[0] < 3 or count[1] < 3):
        raise ValueError(f"Invalid sphere count {count}, must be >= 3")

    sphere = trimesh.creation.uv_sphere(radius=radius, count=count)
    color_uint8 = np.array(color, dtype=np.uint8)

    sphere.visual.face_colors = np.repeat(color_uint8[None, :], len(sphere.faces), axis=0)
    sphere.apply_translation(center)

    scene.add_geometry(sphere)
    return scene


def show_ellipsoid(
    center: np.ndarray,
    radii: np.ndarray,
    color: RGBorRGBAType,
    scene: Optional[trimesh.Scene] = None,
    count: Optional[Tuple[int, int]] = None,
) -> trimesh.Scene:
    """Add an ellipsoid to the scene.

    Args:
        center: ellipsoid center coordinates (3).
        radii: ellipsoid radii (3).
        color: RGB ellipsoid color in 0...255 (3).
        scene: scene to add ellipsoid to.
        count: ellipsoid resolution as number of latitude and longitude lines (2).
    """
    if scene is None:
        scene = trimesh.Scene()
    if np.any(radii < 0.01):
        raise ValueError(f"Invalid ellipsoid radii {radii}, must be > 0.01")
    if radii.shape != (3,):
        raise ValueError(f"Invalid ellipsoid radii shape, expected (3,), got {radii.shape}")
    if count is not None and (count[0] < 3 or count[1] < 3):
        raise ValueError(f"Invalid ellipsoid count {count}, must be >= 3")

    if count is None:
        count = (20, 20)
    nu, nv = count

    # Compute vertices for the ellipsoid using the ellipsoid parametric equation.
    vertices = np.empty(((nu + 1) * (nv + 1), 3))
    for i in range(nu + 1):
        theta = i * np.pi / nv  # angle for the latitude (0 to pi)
        for j in range(nv + 1):
            phi = j * 2 * np.pi / nv  # angle for the longitude (0 to 2*pi)
            x = radii[0] * np.sin(theta) * np.cos(phi)
            y = radii[1] * np.sin(theta) * np.sin(phi)
            z = radii[2] * np.cos(theta)
            vertices[i * (nv + 1) + j] = [x, y, z] + center

    # Compute faces for the ellipsoid.
    faces = np.empty((2 * nu * nv, 3), dtype=int)
    for i in range(nu):
        for j in range(nv):
            v1 = i * (nv + 1) + j
            v2 = (i + 1) * (nv + 1) + j
            v3 = (i + 1) * (nv + 1) + (j + 1)
            v4 = i * (nv + 1) + (j + 1)
            faces[2 * (i * nv + j)] = [v1, v2, v3]
            faces[2 * (i * nv + j) + 1] = [v1, v3, v4]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    scene.add_geometry(mesh)
    return scene


def show_capsule(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    color: RGBorRGBAType,
    scene: Optional[trimesh.Scene] = None,
    count: Optional[Tuple[int, int]] = None,
) -> trimesh.Scene:
    """Add a capsule to the scene.

    Args:
        p1: capsule start point (3).
        p2: capsule end point (3).
        radius: capsule radius.
        color: RGB capsule color in 0...255 (3).
        scene: scene to add capsule to.
        count: sphere resolution as number of latitude and longitude lines (2).
    """
    if scene is None:
        scene = trimesh.Scene()
    if radius < 0.01:
        raise ValueError(f"Invalid capsule radius {radius}, must be > 0.01")
    if np.allclose(p1, p2):
        raise ValueError(f"Invalid capsule points {p1}, {p2}, must be different")
    if count is not None and (count[0] < 3 or count[1] < 3):
        raise ValueError(f"Invalid sphere count {count}, must be >= 3")

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
    vx = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    )  # skew-symmetric matrix of v

    Rc = np.eye(3) + vx + np.matmul(vx, vx) * (1 - c) / (s**2 + 1e-6)  # Rodrigues formula
    center = p1 + dp / 2  # equals P

    Tz = np.eye(4)
    Tz[:3, 3] = center
    Tz[:3, :3] = Rc

    capsule_mesh = trimesh.creation.capsule(radius=radius, height=height, transform=Tz, count=count)

    # Add color to capsule.
    color_uint8 = np.array(color, dtype=np.uint8)
    capsule_mesh.visual.face_colors = np.repeat(
        color_uint8[None, :], len(capsule_mesh.faces), axis=0
    )

    scene.add_geometry(capsule_mesh)
    return scene


def show_cylinder(
    T: np.ndarray,
    radius: float,
    height: float,
    color: RGBorRGBAType,
    scene: Optional[trimesh.Scene] = None,
    count: Optional[int] = None,
) -> trimesh.Scene:
    """Add a cylinder to the scene.

    Args:
        T: center pose as transformation matrix (4, 4).
        radius: cylinder radius.
        height: cylinder height (along body z-axis, height / 2 above and below origin).
        color: RGB capsule color in 0...255 (3).
        scene: scene to add capsule to.
        count: How many pie wedges should the cylinder have
    """
    if scene is None:
        scene = trimesh.Scene()
    if radius < 0.01:
        raise ValueError(f"Invalid cylinder radius {radius}, must be > 0.01")
    if height < 0.01:
        raise ValueError(f"Invalid cylinder height {height}, must be > 0.01")
    if count is not None and count < 3:
        raise ValueError(f"Invalid cylinder count {count}, must be >= 3")

    cylinder_mesh = trimesh.creation.cylinder(
        radius=radius, height=height, transform=T, count=count
    )

    # Add color to cylinder.
    color_uint8 = np.array(color, dtype=np.uint8)
    cylinder_mesh.visual.face_colors = np.repeat(
        color_uint8[None, :], len(cylinder_mesh.faces), axis=0
    )

    scene.add_geometry(cylinder_mesh)
    return scene
