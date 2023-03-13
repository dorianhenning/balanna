import numpy as np
import time

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import Qt
from balanna.display_scenes import display_real_time


# Example: https://realpython.com/python-pyqt-qthread/
class DemoNode(QObject):
    finished = pyqtSignal()
    scene_dict = pyqtSignal(dict)

    def run(self):
        """Long-running task."""
        for i in range(5):
            print(i)
            time.sleep(0.1)
            image = (np.random.random((3, 240, 240)) * 255).astype(np.uint8)
            self.scene_dict.emit({"image": image})
        self.finished.emit()


if __name__ == '__main__':
    worker = DemoNode()
    display_real_time(worker, image_keys=["image"])
