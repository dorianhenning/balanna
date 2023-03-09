import numpy as np

from balanna.trimesh import show_point_cloud, show_grid
from balanna.display_scenes import display_scenes


def main():
    pcs = np.random.rand(10, 100, 3) * 3
    for ts in range(10):
        print(f'current timestamp: {ts}')
        scene = show_point_cloud(pcs[ts])
        scene = show_grid(-100, 100, alpha=100, scene=scene)
        yield {'point_cloud': scene}


if __name__ == '__main__':
    display_scenes(main(), fps=2)
