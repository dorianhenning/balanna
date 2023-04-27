import numpy as np
import trimesh
import trimesh.creation

from balanna.trimesh import show_grid
from balanna import display_generated


def main():
    for k in range(20):
        scene1 = trimesh.Scene()
        box_mesh_1 = trimesh.creation.box(bounds=[(-1, -1, 0), (1, 1, 1)])
        box_mesh_1.visual.vertex_colors = (255, 0, 0)
        T1 = np.eye(4)
        T1[0, 3] = k * 0.2
        box_mesh_1 = box_mesh_1.apply_transform(T1)
        scene1.add_geometry(box_mesh_1)
        scene1 = show_grid(scene=scene1)

        scene2 = trimesh.Scene()
        box_mesh_2 = trimesh.creation.box(bounds=[(-1, -1, 0), (1, 1, 1)])
        box_mesh_2.visual.vertex_colors = (0, 255, 0)
        scene2.add_geometry(box_mesh_2)
        scene2 = show_grid(scene=scene2)

        yield {"scene1": scene1, "scene2": scene2}


if __name__ == '__main__':
    display_generated(main(), fps=10)
