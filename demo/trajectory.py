import numpy as np
import trimesh
import trimesh.creation
import trimesh.path
import trimesh.path.entities

from balanna.trimesh import show_grid, show_trajectory
from balanna import display_generated


def main():
    t = 40
    trajectory = np.stack([np.linspace(0, 10.0, t), np.zeros(t), np.ones(t) * 0.05], axis=1)
    trajectory[:, 1] += np.random.random(t)  # noise

    for k in range(t):
        scene = trimesh.Scene()

        box_mesh = trimesh.creation.box(bounds=[(-0.1, -0.1, 0), (0.1, 0.1, 0.1)])
        T = np.eye(4)
        T[:2, 3] = trajectory[k, :2]
        box_mesh = box_mesh.apply_transform(T)
        scene.add_geometry(box_mesh)

        scene = show_grid(-20, 20, scene=scene)
        scene = show_trajectory(trajectory[:k+1], fade_out=True, scene=scene)
        yield {"scene": scene}


if __name__ == '__main__':
    display_generated(main(), fps=10)
