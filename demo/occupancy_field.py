import numpy as np
import vedo

from balanna import display_generated
from balanna.components import show_axes, show_grid


def main():
    num_time_steps = 50
    resolution = 0.1

    T_W_B = np.stack([np.eye(4) for _ in range(num_time_steps)], axis=0)
    T_W_B[:, 0, 3] = np.linspace(0, 0.9, num=num_time_steps)
    T_W_B[:, 1, 3] = 0.5
    T_W_B[:, 2, 3] = 0.5

    num_cells = int(1.0 / resolution)
    occ_grid = np.zeros((num_time_steps, num_cells, num_cells, num_cells), dtype=np.float32)
    for k in range(num_time_steps):
        gx, gy, gz = tuple(np.round(T_W_B[k, :3, 3] / resolution).astype(int))
        occ_grid[k, gx, gy, gz] = 1.0

    for k in range(num_time_steps):
        scene = show_axes([T_W_B[k], np.eye(4)])
        scene = show_grid(scene=scene)
        yield {'scene': scene, 'occupancy': vedo.Volume(occ_grid[k], alpha=(0.0, 0.5, 1.0))}


if __name__ == '__main__':
    display_generated(main())
