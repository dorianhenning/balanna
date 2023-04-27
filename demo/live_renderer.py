import numpy as np
import time
import trimesh
import trimesh.creation

from balanna.display_scenes import display_real_time, RealTimeNode
from balanna.trimesh import show_grid


class DemoNode(RealTimeNode):

    def __init__(self, parent=None):
        super(RealTimeNode, self).__init__(parent=parent)
        self.k = 0

    def callback(self):
        T = np.eye(4)
        T[0, 3] = self.k * 0.2
        image = (np.random.random((3, 240, 240)) * 255).astype(np.uint8)

        scene = trimesh.Scene()
        box_mesh = trimesh.creation.box(bounds=[(-1, -1, 0), (1, 1, 1)])
        box_mesh = box_mesh.apply_transform(T)
        scene.add_geometry(box_mesh)
        scene = show_grid(xy_max=20, scene=scene)

        self.render({"image": image, "scene": scene})

        self.k += 1
        time.sleep(0.1)
        if self.k > 500:
            self.close()


if __name__ == '__main__':
    worker = DemoNode()
    display_real_time(worker, image_keys=["image"], scene_keys=["scene"])
