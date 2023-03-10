import numpy as np

from typing import Tuple, Union


def create_round_flight(
    timestamps: np.ndarray,
    focal_point: Union[Tuple[float, float, float], np.ndarray] = (0, 0, 0),
    cam_height: float = 2.0,
    radius: float = 5.0
):
    """Create camera trajectory circling around an object in the focal point, in the xy-plane on z = cam_height
    and with the given radius.

    Args:
        timestamps: trajectory timestamps in radians (N,) -> 2*pi = full circle.
        focal_point: trajectory focal point, i.e. each camera points at this point.
        cam_height: camera z coordinate.
        radius: circle radius.
    Returns:
        T_W_C: world to camera transforms (N, 4, 4).
    """
    if len(timestamps.shape) != 1:
        raise ValueError(f"Expected flat timestamp vector, got {timestamps.shape}")

    num_time_stamps = len(timestamps)
    cam_t = np.stack([focal_point[0] + radius * np.cos(timestamps),
                      focal_point[1] + radius * np.sin(timestamps),
                      focal_point[2] + np.ones_like(timestamps) * cam_height], axis=1)

    T_W_C = np.zeros((num_time_stamps, 4, 4))
    focal_point_ = np.array(focal_point)
    for k in range(num_time_stamps):
        cam_z = focal_point_ - cam_t[k]
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.cross(np.array([0, 0, 1]), cam_t[k])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.cross(cam_z, cam_x)

        T_W_C[k] = np.eye(4)
        T_W_C[k, :3, 3] = cam_t[k]
        T_W_C[k, :3, :3] = np.stack([cam_x, cam_y, cam_z]).T

    return T_W_C
