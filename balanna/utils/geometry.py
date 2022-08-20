import numpy as np
import trimesh

from typing import List, Dict


def quaternion_to_trimesh_format(quaternion: np.ndarray, q_order: str = "XYZW"):
    if q_order == "XYZW":
        return quaternion[..., [3, 0, 1, 2]]
    elif q_order == "WXYZ":
        return quaternion
    raise ValueError(f"Unknown quaternion config {q_order}")


def pose_to_trimesh_tf(position: np.ndarray, quaternion: np.ndarray, q_order: str = "XYZW"):
    q = quaternion_to_trimesh_format(quaternion, q_order=q_order)
    T = trimesh.transformations.quaternion_matrix(q)
    T[..., :3, 3] = position
    return T


def to_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


def transform_to_vtkcamera(transform: np.ndarray) -> Dict[str, List[float]]:
    if transform.shape != (4, 4):
        raise ValueError(f"Expected single transformation matrix, got {transform.shape}")
    position = transform[:3, 3]
    z_direction = transform[2, :3]
    y_direction = transform[1, :3]
    focal_point = position - z_direction

    return {'pos': position.tolist(),
            'focalPoint': focal_point.tolist(),
            'viewup': y_direction.tolist()}
