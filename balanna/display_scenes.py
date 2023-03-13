import datetime
import image_grid
import importlib.metadata
import numpy as np
import packaging.version
import pathlib
import trimesh.path.entities
import trimesh.visual
import trimesh.viewer
import time
import vedo

from functools import partial
from PIL import Image, ImageQt
from PyQt5 import Qt
from typing import Any, Dict, Iterable, Optional, Union
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


SceneDictType = Dict[str, Any]


class MainWindow(Qt.QMainWindow):

    def __init__(
        self,
        scene_iterator: Iterable[SceneDictType],
        fps: float,
        video_directory: Optional[Union[pathlib.Path, str]] = None,
        horizontal: bool = True,
        loop: bool = True,
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
        self.scene_iterator = scene_iterator
        self.fps = fps
        self.show_labels = show_labels
        self.use_scene_cam = use_scene_cam
        self.debug = debug
        scene_dict = self.get_next_scene_dict()
        if scene_dict is None:
            return

        image_keys = [key for key, value in scene_dict.items() if isinstance(value, np.ndarray)]
        vl1 = Qt.QHBoxLayout()
        self.image_widge_dict = dict()
        for key in image_keys:
            self.image_widge_dict[key] = Qt.QLabel()
            vl1.addWidget(self.image_widge_dict[key])
        vl.addLayout(vl1)

        scene_keys = [key for key, value in scene_dict.items() if isinstance(value, trimesh.Scene)]
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
            vp.add_callback('KeyPress', self.__on_key)
            vp.show()
            self.vtkWidgets.append(vtk_widget)
            self.vps.append(vp)
        vl.addLayout(vl2)

        self._mouse_timer = Qt.QTimer(self)
        self._mouse_timer.timeout.connect(self.__align_event)
        self._current_align_id = None

        frame.setLayout(vl)
        self.setCentralWidget(frame)
        self.show()

        self.video_directory = None
        self.__video_index = 0
        if video_directory is not None:
            self.video_directory = pathlib.Path(video_directory)
            if self.video_directory.exists():
                print("\033[93m" + "Video directory already exists, overwriting it ..." + "\033[0m")

        self.render_(scene_dict, resetcam=True)
        self.timer = Qt.QTimer(self)
        self.timer.timeout.connect(self.render_next_scene)
        if self.debug:
            print("\033[36m" + "Setup ready, starting to loop ..." + "\033[0m")
        if loop:
            self.toggle_looping()

    def render_next_scene(self, resetcam: bool = False) -> None:
        scene_dict = self.get_next_scene_dict()
        if scene_dict is not None:
            self.render_(scene_dict, resetcam=resetcam)
        elif self.timer.isActive():
            self.timer.stop()

    def render_(self, scene_dict: SceneDictType, resetcam: bool = False):
        start_time = time.perf_counter()

        for key, element in scene_dict.items():
            if isinstance(element, trimesh.Scene) and key in self.scene_key_dict:
                at = self.scene_key_dict[key]
                meshes_vedo = []
                for m in element.geometry.values():
                    if isinstance(m, trimesh.Trimesh):
                        m_vedo = vedo.trimesh2vedo(m)
                        if m.visual.kind == "vertex":
                            face_colors = trimesh.visual.color.vertex_to_face_color(m.visual.vertex_colors, m.faces)
                            m_vedo.cellIndividualColors(face_colors)
                        meshes_vedo.append(m_vedo)

                    elif isinstance(m, trimesh.PointCloud):
                        m_vedo = vedo.Points(m.vertices, c=m.visual.vertex_colors)
                        meshes_vedo.append(m_vedo)

                    elif isinstance(m, trimesh.path.Path3D):
                        # The trimesh path consists of entities and vertices. The vertices are the 3D points,
                        # that are connected as described in the entities.
                        if not all([isinstance(me, trimesh.path.entities.Line) for me in m.entities]):
                            raise ValueError("Currently only trimesh.path.entities.Line entities are supported")
                        if not all([len(me.points) == 2 for me in m.entities]):
                            raise ValueError("Invalid line entities, should have point lists [start, end]")

                        # Add each line segment individually as a vedo line to support multicolored lines
                        # and different alpha values along the line.
                        for ke, line_entity in enumerate(m.entities):
                            i, j = line_entity.points
                            c = m.colors[ke, :3]
                            alpha = m.colors[ke, -1] / 255  # [0, 255] -> [0, 1]
                            m_vedo = vedo.Lines(m.vertices[None, i], m.vertices[None, j], lw=2, c=c, alpha=alpha)
                            meshes_vedo.append(m_vedo)

                    else:
                        meshes_vedo.append(m)

                if self.show_labels:
                    annotation = vedo.CornerAnnotation()
                    annotation.text(key)
                    meshes_vedo.append(annotation)

                camera_dict = None
                if self.use_scene_cam:
                    focal_distance = 1.0
                    T_W_C = element.camera_transform
                    cam_0 = T_W_C[:3, 3]
                    cam_1 = cam_0 + T_W_C[:3, :3] @ np.array([0, 0, focal_distance])  # along z of camera
                    view_up = - T_W_C[:3, 1]  # camera convention -> y points down
                    camera_dict = dict(pos=cam_0, focal_point=cam_1, viewup=view_up)

                self.vps[at].clear()
                self.vps[at].show(meshes_vedo, bg="white", resetcam=resetcam, camera=camera_dict)

            elif isinstance(element, np.ndarray) and key in self.image_widge_dict:
                if len(element.shape) == 3:
                    image_mode = "RGB"
                    element = np.transpose(element, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                else:
                    image_mode = "L"
                img = Image.fromarray(element, mode=image_mode)
                qt_img = ImageQt.ImageQt(img)
                self.image_widge_dict[key].setPixmap(Qt.QPixmap.fromImage(qt_img))

            elif isinstance(element, str):
                print(f"{key}: {element}")

            else:
                continue

        if self.debug:
            dt = int((time.perf_counter() - start_time) * 1000)  # secs -> milliseconds
            print("\033[36m" + f"Rendering time: {dt} ms" + "\033[0m")

        if self.video_directory is not None:
            self.screenshot(self.video_directory, prefix=f"{self.__video_index:05d}")
            self.__video_index += 1

    def __align_event(self, event=None, align_id: int = None):  # noqa
        if self.num_scenes < 2:
            return
        if align_id is None:
            align_id = self._current_align_id

        camera = self.vps[align_id].camera
        for k in range(self.num_scenes):
            self.vps[k].renderer.SetActiveCamera(camera)
            self.vps[k].render()

    def __on_key(self, event_dict) -> None:
        key_pressed = event_dict["keyPressed"]
        if key_pressed == "s":
            self.toggle_looping()
        elif key_pressed == "n":
            self.render_next_scene()
        elif key_pressed == "z":
            if self.num_scenes > 0:
                self.vps[0].render(resetcam=True)
                self.__align_event(align_id=0)
        elif key_pressed == "k":
            screenshot_dir = pathlib.Path("screenshots")
            prefix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            self.screenshot(screenshot_dir, prefix=prefix)
        elif key_pressed == "h":
            print("Usage:",
                  "\n\tq: quit",
                  "\n\ts: play / pause",
                  "\n\tn: next",
                  "\n\tz: reset view",
                  "\n\tk: take screenshot")
        elif key_pressed == "q":
            self.close()

    def __on_mouse_start(self, _, align_id: int):
        self._current_align_id = align_id
        self._mouse_timer.start(10)  # in milliseconds

    def __on_mouse_end(self, _):
        if self._mouse_timer.isActive():
            self._mouse_timer.stop()
        self._current_align_id = None

    def get_next_scene_dict(self) -> Optional[SceneDictType]:
        try:
            return next(self.scene_iterator)  # noqa
        except StopIteration:
            return None

    def toggle_looping(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(int(1 / self.fps * 1000))  # in milliseconds

    def screenshot(self, directory: pathlib.Path, prefix: str):
        image_paths = []

        # Store the images from the widget's pixmaps, store them and add them to
        # the list of screenshot files.
        for key, widget in self.image_widge_dict.items():
            path = directory / key / f"{prefix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            widget.pixmap().toImage().save(path)
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


def display_scenes(
    scene_iterator: Iterable[SceneDictType],
    horizontal: bool = True,
    loop: bool = False,
    fps: float = 30.0,
    video_directory: Optional[pathlib.Path] = None,
    show_labels: bool = False,
    use_scene_cam: bool = False,
    debug: bool = False
):
    """Display scenes stored in scene iterator as PyQt app.

    The scene iterator yields a dictionary that describes the elements of the scene, one dictionary per frame.
    Currently, images (np.array), 3D scenes (trimesh.Scene) and strings (for prinout) are supported.

    Args:
        scene_iterator: iterator function to get the scene dictionaries.
        horizontal: window orientation, horizontal or vertical stacking.
        loop: start looping from beginning.
        fps: frames (i.e. scenes) per second for looping.
        video_directory: directory for storing screenshots.
        show_labels: display the scene dict keys.
        use_scene_cam: use camera transform from trimesh scenes.
        debug: printing debug information.
    """
    app = Qt.QApplication([])
    window = MainWindow(
        scene_iterator=scene_iterator,
        fps=fps,
        video_directory=video_directory,
        horizontal=horizontal,
        loop=loop,
        show_labels=show_labels,
        debug=debug,
        use_scene_cam=use_scene_cam
    )
    app.aboutToQuit.connect(window.on_close)
    app.exec_()
