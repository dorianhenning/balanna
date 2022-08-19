import numpy as np
import trimesh.viewer
import vedo

from PIL import Image, ImageQt
from PyQt5 import Qt
from typing import Any, Dict, Iterable, Optional
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


SceneDictType = Dict[str, Any]


class MainWindow(Qt.QMainWindow):

    def __init__(self, scene_iterator: Iterable[SceneDictType], fps: float, parent: Qt.QWidget = None):
        Qt.QMainWindow.__init__(self, parent)
        frame = Qt.QFrame()
        vl = Qt.QVBoxLayout()
        self.scene_iterator = scene_iterator
        self.fps = fps
        scene_dict = self.get_next_scene_dict()
        if scene_dict is None:
            return

        image_keys = [key for key, value in scene_dict.items() if isinstance(value, np.ndarray)]
        vl1 = Qt.QHBoxLayout()
        self.image_frame_dict = dict()
        for key in image_keys:
            self.image_frame_dict[key] = Qt.QLabel()
            vl1.addWidget(self.image_frame_dict[key])
        vl.addLayout(vl1)

        scene_keys = [key for key, value in scene_dict.items() if isinstance(value, trimesh.Scene)]
        self.scene_key_dict = {key: i for i, key in enumerate(scene_keys)}
        self.vtkWidget = QVTKRenderWindowInteractor(frame)
        vl2 = Qt.QHBoxLayout()
        vl2.addWidget(self.vtkWidget)
        self.vp = vedo.Plotter(qtWidget=self.vtkWidget, N=len(scene_keys))
        self.vp.addCallback('KeyPress', self.on_key)
        self.vp.show()
        vl.addLayout(vl2)

        frame.setLayout(vl)
        self.setCentralWidget(frame)
        self.show()

        self.render_(scene_dict, resetcam=True)
        self.timer = Qt.QTimer(self)
        self.timer.timeout.connect(self.render_next_scene)

    def render_next_scene(self, resetcam: bool = False) -> None:
        scene_dict = self.get_next_scene_dict()
        if scene_dict is not None:
            self.render_(scene_dict, resetcam=resetcam)
        elif self.timer.isActive():
            self.timer.stop()

    def render_(self, scene_dict: SceneDictType, resetcam: bool = False):
        for key, element in scene_dict.items():
            if isinstance(element, trimesh.Scene) and key in self.scene_key_dict:
                at = self.scene_key_dict[key]
                meshes_vedo = [vedo.trimesh2vedo(m) for m in element.geometry.values()]
                self.vp.clear(at=at)
                self.vp.show(meshes_vedo, at=at, bg="white", resetcam=resetcam)

            elif isinstance(element, np.ndarray) and key in self.image_frame_dict:
                image_mode = "RGB" if len(element.shape) == 3 else "L"
                img = Image.fromarray(element, mode=image_mode)
                qt_img = ImageQt.ImageQt(img)
                self.image_frame_dict[key].setPixmap(Qt.QPixmap.fromImage(qt_img))

            else:
                continue

    def on_key(self, event_dict) -> None:
        key_pressed = event_dict["keyPressed"]
        if key_pressed == "s":
            if self.timer.isActive():
                self.timer.stop()
            else:
                self.timer.start(int(1 / self.fps * 1000))  # in milliseconds
        elif key_pressed == "n":
            self.render_next_scene()
        elif key_pressed == "z":
            self.vp.render(resetcam=True)
        elif key_pressed == "h":
            print("Usage:",
                  "\n\tq: quit",
                  "\n\ts: play / pause",
                  "\n\tn: next",
                  "\n\tz: reset view")
        elif key_pressed == "q":
            self.close()

    def get_next_scene_dict(self) -> Optional[SceneDictType]:
        try:
            return next(self.scene_iterator)
        except StopIteration:
            return None

    def on_close(self):
        self.vtkWidget.close()
        for key, widget in self.image_frame_dict.items():
            widget.close()


def display_scenes(scene_iterator: Iterable[SceneDictType], fps: float = 30.0):
    app = Qt.QApplication([])
    window = MainWindow(scene_iterator, fps=fps)
    app.aboutToQuit.connect(window.on_close)
    app.exec_()
