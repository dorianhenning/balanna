import numpy as np
import trimesh.viewer
import vedo

from PIL import Image, ImageQt
from PyQt5 import Qt
from typing import Any, Dict, Iterable, Optional, Tuple
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


SceneDictType = Dict[str, Any]


class MainWindow(Qt.QMainWindow):

    def __init__(self, scene_iterator: Iterable[SceneDictType], fps: float, horizontal: bool = True):
        Qt.QMainWindow.__init__(self)
        frame = Qt.QFrame()
        if horizontal:
            vl = Qt.QHBoxLayout()
        else:
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

        self.scene_dict = scene_dict
        self.timer = Qt.QTimer(self)
        self.timer.timeout.connect(self.render_next_scene)

        window_size = Qt.QSize(self.num_widgets * 200, 200) if horizontal else Qt.QSize(200, self.num_widgets * 200)
        self.resize(window_size)
        self.render_(scene_dict, resetcam=True, )
        self.show()

    def resizeEvent(self, a0: Qt.QResizeEvent) -> None:
        dw = int(a0.size().width() / self.num_widgets)
        dh = int(a0.size().height() / self.num_widgets)
        self.render_(self.scene_dict, widget_size=(dw, dh))

    def render_next_scene(self, resetcam: bool = False) -> None:
        scene_dict = self.get_next_scene_dict()
        if scene_dict is not None:
            self.scene_dict = scene_dict
            self.render_(scene_dict, resetcam=resetcam)
        elif self.timer.isActive():
            self.timer.stop()

    def render_(self, scene_dict: SceneDictType, resetcam: bool = False, widget_size: Optional[Tuple[int, int]] = None):
        for key, element in scene_dict.items():
            if isinstance(element, trimesh.Scene) and key in self.scene_key_dict:
                at = self.scene_key_dict[key]
                meshes_vedo = [vedo.trimesh2vedo(m) if isinstance(m, trimesh.Trimesh) else m
                               for m in element.geometry.values()]
                self.vp.clear(at=at)
                self.vp.show(meshes_vedo, at=at, bg="white", resetcam=resetcam, size=widget_size)

            elif isinstance(element, np.ndarray) and key in self.image_frame_dict:
                image_mode = "RGB" if len(element.shape) == 3 else "L"
                img = Image.fromarray(element, mode=image_mode)
                qt_img = ImageQt.ImageQt(img)
                pixmap = Qt.QPixmap.fromImage(qt_img)
                if widget_size is not None:
                    pixmap = pixmap.scaled(*widget_size)
                self.image_frame_dict[key].setPixmap(pixmap)

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

    @property
    def num_widgets(self) -> int:
        return len(self.centralWidget().layout().children())


def display_scenes(scene_iterator: Iterable[SceneDictType], horizontal: bool = True, fps: float = 30.0):
    app = Qt.QApplication([])
    window = MainWindow(
        scene_iterator=scene_iterator,
        fps=fps,
        horizontal=horizontal,
    )
    app.aboutToQuit.connect(window.on_close)
    app.exec_()
