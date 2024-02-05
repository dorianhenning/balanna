import numpy as np
import trimesh
import trimesh.creation

from balanna.trimesh import show_trajectory
from balanna import display_generated


def main():
    scene = trimesh.Scene()

    # Different trajectories.
    points = np.zeros((100, 3))
    points[:, 0] = np.linspace(-1, 1, 100)
    points[:, 1] = np.sin(points[:, 0] * np.pi)
    
    points[:, 2] = 0
    scene = show_trajectory(points, (255, 0, 0), scene=scene)
    points[:, 2] = 1
    scene = show_trajectory(points, (255, 0, 0), fade_out=True, scene=scene)
    points[:, 2] = 2
    scene = show_trajectory(points, (255, 0, 0, 100), scene=scene)
    points[:, 2] = 3
    colors = np.random.randint(0, 255, (99, 3), dtype=np.uint8)
    scene = show_trajectory(points, colors, scene=scene)

    yield {"scene": scene}


if __name__ == '__main__':
    display_generated(main(), fps=10)
