import numpy as np

from balanna.camera_trajectories import create_round_flight
from balanna.trimesh import show_point_cloud, show_grid, show_camera
from balanna.display_scenes import display_scenes


def main():
    num_time_steps = 50
    pcs = np.random.rand(num_time_steps, 307200, 3) * 3 - 1.5
    colors = (np.random.rand(num_time_steps, 307200, 3) * 255).astype(np.uint8)
    t = np.linspace(0, 10, num_time_steps)
    T_W_C = create_round_flight(t, focal_point=(0, 0, 0), radius=3.0)

    for k in range(num_time_steps):
        scene = show_point_cloud(pcs[k], colors=colors[k])
        scene = show_grid(-10, 10, alpha=100, scene=scene)
        scene = show_camera(T_W_C[k], scene=scene)
        yield {'scene': scene}


if __name__ == '__main__':
    display_scenes(main())
