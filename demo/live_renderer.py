import numpy as np
import time
import trimesh
import trimesh.creation

from PyQt5.QtCore import QObject, pyqtSignal
from balanna.display_scenes import display_real_time
from balanna.trimesh import show_grid


class DemoNode(QObject):
    finished = pyqtSignal()
    running = pyqtSignal()
    scene_dict = pyqtSignal(dict)

    def run(self):
        for k in range(500):
            time.sleep(0.1)
            self.callback(k)
        self.finished.emit()  # noqa

    def callback(self, k: int):
        if not self.running:
            return
        scene = trimesh.Scene()
        box_mesh = trimesh.creation.box(bounds=[(-1, -1, 0), (1, 1, 1)])
        T = np.eye(4)
        T[0, 3] = k * 0.2
        box_mesh = box_mesh.apply_transform(T)
        scene.add_geometry(box_mesh)
        scene = show_grid(scene=scene)

        image = (np.random.random((3, 240, 240)) * 255).astype(np.uint8)

        self.scene_dict.emit({"image": image, "scene": scene})  # noqa


if __name__ == '__main__':
    worker = DemoNode()
    display_real_time(worker, image_keys=["image"], scene_keys=["scene"])
