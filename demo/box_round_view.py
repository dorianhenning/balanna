import numpy as np
import trimesh
import trimesh.creation

from balanna.trimesh import show_grid
from balanna.display_scenes import display_scenes


def main():
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(bounds=[(-1, -1, 0),
                                                    (1, 1, 1)]))
    scene = show_grid(-100, 100, alpha=100, scene=scene)

    num_time_steps = 50
    t = np.linspace(0, 10, num_time_steps)
    cam_t = np.stack([10.0 * np.cos(t), 10.0 * np.sin(t), np.ones_like(t) * 2], axis=1)
    for k in range(num_time_steps):
        T_W_C = np.eye(4)
        T_W_C[:3, 3] = cam_t[k]
        scene.camera_transform = T_W_C
        yield {'scene': scene}


if __name__ == '__main__':
    display_scenes(main(), fps=10, use_scene_cam=True)
