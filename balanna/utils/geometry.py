import numpy as np
import trimesh


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
