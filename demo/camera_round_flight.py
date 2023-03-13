import numpy as np
import trimesh
import trimesh.creation
import trimesh.path

from balanna.camera_trajectories import create_round_flight
from balanna.trimesh import show_grid
from balanna.display_scenes import display_scenes


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
