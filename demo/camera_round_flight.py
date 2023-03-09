import numpy as np
import trimesh
import trimesh.creation
import trimesh.path

from balanna.trimesh import show_grid
from balanna.display_scenes import display_scenes


def create_round_flight(t: np.ndarray, radius: float):
    num_time_stamps = len(t)
    assert t.shape == (num_time_stamps, )
    cam_t = np.stack([radius * np.cos(t), radius * np.sin(t), np.ones_like(t) * 2], axis=1)

    T_W_C = np.zeros((num_time_stamps, 4, 4))
    for k in range(num_time_stamps):
        cam_z = np.zeros(3) - cam_t[k]  # focal point = (0, 0, 0)
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.cross(np.array([0, 0, 1]), cam_t[k])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.cross(cam_z, cam_x)

        T_W_C[k] = np.eye(4)
        T_W_C[k, :3, 3] = cam_t[k]
        T_W_C[k, :3, :3] = np.stack([cam_x, cam_y, cam_z]).T

    return T_W_C


def main():
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(bounds=[(-1, -1, 0),
                                                    (1, 1, 1)]))
    scene = show_grid(-10, 10, alpha=100, scene=scene)

    num_time_steps = 50
    t = np.linspace(0, 10, num_time_steps)
    T_W_C = create_round_flight(t, radius=10.0)
    for k in range(num_time_steps):
        scene.camera_transform = T_W_C[k]
        yield {'scene': scene}


if __name__ == '__main__':
    display_scenes(main(), fps=10, use_scene_cam=True)
