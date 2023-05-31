import datetime
import image_grid
import importlib.metadata
import numpy as np
import packaging.version
import pathlib
import trimesh.path.entities
import trimesh.viewer
import time
import vedo

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from functools import partial
from PIL import Image
from PyQt5 import Qt
from typing import Any, Dict, List, Optional, Union
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from balanna.utils.vedo_utils import trimesh_scene_2_vedo

SceneDictType = Dict[str, Union[trimesh.Scene, Axes, str, np.ndarray, vedo.Volume]]


class MainWindow(Qt.QMainWindow):

    def __init__(
        self,
        image_keys: List[str],
        scene_keys: List[str],
        figure_keys: List[str],
        video_directory: Optional[Union[pathlib.Path, str]] = None,
        horizontal: bool = True,
        show_labels: bool = False,
        use_scene_cam: bool = False,
        debug: bool = False,
        parent: Qt.QWidget = None
    ):
        Qt.QMainWindow.__init__(self, parent)
        frame = Qt.QFrame()
        if horizontal:
            vl = Qt.QHBoxLayout()
        else:
            vl = Qt.QVBoxLayout()
        self.show_labels = show_labels
        self.use_scene_cam = use_scene_cam
        self.debug = debug

        # Set up image-based widgets (i.e. numpy array inputs).
        vl1 = Qt.QHBoxLayout()
        self.image_widge_dict = dict()
        for key in image_keys:
            self.image_widge_dict[key] = Qt.QLabel()
            vl1.addWidget(self.image_widge_dict[key])
        vl.addLayout(vl1)

        # Set up 3D scene widgets (i.e. trimesh scenes).
        self.scene_key_dict = {key: i for i, key in enumerate(scene_keys)}
        self.vps = []
        self.vtkWidgets = []
        vl2 = Qt.QHBoxLayout()
        for _, scene_id in self.scene_key_dict.items():
            vtk_widget = QVTKRenderWindowInteractor(frame)
            vl2.addWidget(vtk_widget)
            vedo_version = importlib.metadata.version('vedo')
            if packaging.version.parse(vedo_version) > packaging.version.parse('2022.3.0'):
                vp = vedo.Plotter(qt_widget=vtk_widget)
            else:
                vp = vedo.Plotter(qtWidget=vtk_widget)  # noqa
            vp.add_callback('LeftButtonPress', partial(self.__on_mouse_start, align_id=scene_id))
            vp.add_callback('LeftButtonRelease', self.__on_mouse_end)
            vp.add_callback('MiddleButtonPress', partial(self.__on_mouse_start, align_id=scene_id))
            vp.add_callback('MiddleButtonRelease', self.__on_mouse_end)
            vp.add_callback('MouseWheelForward', partial(self.__align_event, align_id=scene_id))
            vp.add_callback('MouseWheelBackward', partial(self.__align_event, align_id=scene_id))
            vp.add_callback('KeyPress', self._on_key)
            vp.show()
            self.vtkWidgets.append(vtk_widget)
            self.vps.append(vp)
        vl.addLayout(vl2)

        vl3 = Qt.QVBoxLayout()
        self.figure_key_dict = {key: i for i, key in enumerate(figure_keys)}
        self.figureCanvas = []
        for _ in figure_keys:
            figure_canvas = FigureCanvas(Figure())
            self.figureCanvas.append(figure_canvas)
            vl3.addWidget(figure_canvas)
        vl.addLayout(vl3)

        # Set up mouse timer for synchronizing mouse interactions across multiple (3D) widgets.
        # Therefore, the interaction is detected, tracked and transferred to all other 3D widgets.
        self._mouse_timer = Qt.QTimer(self)
        self._mouse_timer.timeout.connect(self.__align_event)  # noqa
        self._current_align_id = None

        frame.setLayout(vl)
        self.setCentralWidget(frame)
        # If image and scene keys are displayed, somehow the scene widgets start with zero width.
        # Therefore, resize the window before startup by multiplying the current size with the
        # number of scene widgets.
        if len(image_keys) > 0 and len(scene_keys) > 0:
            w = self.size().width() + self.size().width() * len(scene_keys)
            h = self.size().height()
            self.resize(w, h)
        self.show()

        # Set up video directory for storing screenshots.
        self.video_directory = None
        self.__video_index = 0
        if video_directory is not None:
            self.video_directory = pathlib.Path(video_directory)
            if self.video_directory.exists():
                print("\033[93m" + "Video directory already exists, overwriting it ..." + "\033[0m")
            self.video_directory.mkdir(exist_ok=True, parents=True)

    def render_(self, scene_dict: SceneDictType, resetcam: bool = False):
        start_time = time.perf_counter()

        for key, element in scene_dict.items():
            if isinstance(element, trimesh.Scene):
                meshes_vedo, camera_dict = trimesh_scene_2_vedo(
                    scene=element,
                    label=key if self.show_labels else None,
                    use_scene_cam=self.use_scene_cam
                )
                self.render_scene_at_key_(key, meshes_vedo, resetcam=resetcam, camera_dict=camera_dict)

            elif isinstance(element, np.ndarray) and key in self.image_widge_dict:
                if len(element.shape) == 3:
                    if element.shape[0] == 3:
                        element = np.transpose(element, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                else:
                    element = np.stack([element] * 3, axis=-1)
                height, width, _ = element.shape
                qt_img = Qt.QImage(element.data.tobytes(), width, height, 3 * width, Qt.QImage.Format_RGB888)
                self.image_widge_dict[key].setPixmap(Qt.QPixmap.fromImage(qt_img))

            elif isinstance(element, str):
                print(f"{key}: {element}")

            elif isinstance(element, Axes) and key in self.figure_key_dict:
                at = self.figure_key_dict[key]
                # clear figure
                self.figureCanvas[at].figure.clear()
                # set new references of axes element
                element.figure = self.figureCanvas[at].figure
                self.figureCanvas[at].figure.add_axes(element)
                # draw figure to update!
                self.figureCanvas[at].draw()

            elif isinstance(element, vedo.Volume):
                self.render_scene_at_key_(key, [element], resetcam=resetcam)

            else:
                print("\033[93m" + f"Invalid element in scene dict of type {type(element)}, skipping ..." + "\033[0m")
                continue

        if self.debug:
            dt = int((time.perf_counter() - start_time) * 1000)  # secs -> milliseconds
            print("\033[36m" + f"Rendering time: {dt} ms" + "\033[0m")

        if self.video_directory is not None:
            self.screenshot(self.video_directory, prefix=f"{self.__video_index:05d}")
            self.__video_index += 1

    def render_scene_at_key_(
        self,
        key: str,
        meshes: List[Union[vedo.Mesh, vedo.Volume]],
        resetcam: bool = False,
        camera_dict: Optional[Dict[str, np.ndarray]] = None
    ):
        if key not in self.scene_key_dict:
            print("\033[93m" + f"Key {key} not in scene key dicts, skipping ..." + "\033[0m")
            return

        at = self.scene_key_dict[key]
        self.vps[at].clear()
        self.vps[at].show(meshes, bg="white", resetcam=resetcam, camera=camera_dict)

    def __align_event(self, event=None, align_id: int = None):  # noqa
        if self.num_scenes < 2:
            return
        if align_id is None:
            align_id = self._current_align_id

        camera = self.vps[align_id].camera
        for k in range(self.num_scenes):
            self.vps[k].renderer.SetActiveCamera(camera)
            self.vps[k].render()

    def keyPressEvent(self, event):
        """In case the Qt key callback is called, convert the Qt event to a string key and call the _on_key() method.
        However, this case will only occur if the window contains no vedo panel. Thus, duplicated calling of
        the key callback is not an issue.
        """
        event_dict = {"keyPressed": chr(event.key()).lower()}
        self._on_key(event_dict)
        event.accept()

    def _on_key(self, event_dict) -> None:
        key_pressed = event_dict["keyPressed"]
        if key_pressed == "z":
            if self.num_scenes > 0:
                self.vps[0].render(resetcam=True)
                self.__align_event(align_id=0)
        elif key_pressed == "k":
            screenshot_dir = pathlib.Path("screenshots")
            prefix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            self.screenshot(screenshot_dir, prefix=prefix)
        elif key_pressed == "h":
            self.print_usage()
        elif key_pressed == "q":
            self.close()

    def __on_mouse_start(self, _, align_id: int):
        self._current_align_id = align_id
        self._mouse_timer.start(10)  # in milliseconds

    def __on_mouse_end(self, _):
        if self._mouse_timer.isActive():
            self._mouse_timer.stop()
        self._current_align_id = None

    @staticmethod
    def print_usage():
        print("Usage:",
              "\n\tq: quit",
              "\n\tz: reset view",
              "\n\tk: take screenshot")

    def screenshot(self, directory: pathlib.Path, prefix: str):
        image_paths = []

        # Store the images from the widget's pixmaps, store them and add them to
        # the list of screenshot files.
        for key, widget in self.image_widge_dict.items():
            path = directory / key / f"{prefix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            widget.pixmap().toImage().save(path.as_posix())
            image_paths.append(path)

        # Make screenshots of all trimesh scenes, store them and add them to
        # the list of screenshot files.
        for key, widget in self.scene_key_dict.items():
            path = directory / key / f"{prefix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            at = self.scene_key_dict[key]
            self.vps[at].screenshot(path)
            image_paths.append(path)

        # Make to image grid of all screenshot files and store it.
        image_grid_ = image_grid.image_grid(image_paths)[..., :3]  # RGBA -> RGB
        Image.fromarray(image_grid_).save(directory / f"{prefix}__.png")

    def on_close(self):
        for vtk_widget in self.vtkWidgets:
            vtk_widget.close()
        for key, widget in self.image_widge_dict.items():
            widget.close()

    @property
    def num_images(self) -> int:
        return len(self.image_widge_dict)

    @property
    def num_scenes(self) -> int:
        return len(self.vps)
