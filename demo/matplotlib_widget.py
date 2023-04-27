import numpy as np
import trimesh
import trimesh.creation
import matplotlib.pyplot as plt

from balanna.trimesh import show_grid
from balanna import display_scenes


def main():
    for k in range(20):
        scene = trimesh.Scene()
        box_mesh = trimesh.creation.box(bounds=[(-1, -1, 0), (1, 1, 1)])
        box_mesh.visual.vertex_colors = (255, 0, 0)
        T = np.eye(4)
        T[0, 3] = k * 0.2
        box_mesh = box_mesh.apply_transform(T)
        scene.add_geometry(box_mesh)
        scene = show_grid(scene=scene)

        image = np.random.randint(0, 255, (240, 320, 3)).astype(np.uint8)

        figure, axes = plt.subplots()
        t = np.linspace(0, 10, 501)
        axes.plot(t, np.sin(t + k * 0.1))

        yield {"scene": scene, "image": image, "figure": axes}


if __name__ == '__main__':
    display_scenes(main(), fps=10)
