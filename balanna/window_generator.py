import numpy as np
import pathlib

from matplotlib.axes import Axes
from PyQt5 import Qt
from typing import Iterable, Optional, Union, Type

from .window_base import MainWindow, SceneDictType
from .utils.types import contains_scene


__all__ = ['display_scenes', 'display_generated', 'display_generated_custom', 'MainWindowGenerator']


class MainWindowGenerator(MainWindow):

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
        self.scene_iterator = scene_iterator
        self.fps = fps
        scene_dict = self.get_next_scene_dict()
        if scene_dict is None:
            return

        image_keys = [key for key, value in scene_dict.items() if isinstance(value, np.ndarray)]
        scene_keys = [key for key, value in scene_dict.items() if contains_scene(value)]
        figure_keys = [key for key, value in scene_dict.items() if isinstance(value, Axes)]
        super(MainWindowGenerator, self).__init__(
            image_keys=image_keys,
            scene_keys=scene_keys,
            figure_keys=figure_keys,
            video_directory=video_directory,
            horizontal=horizontal,
            show_labels=show_labels,
            use_scene_cam=use_scene_cam,
            debug=debug,
            parent=parent
        )

        self.render_(scene_dict, resetcam=True)

        # Setup looping timer and connect it to render a new scene at every timer callback.
        # If loop is true, start looping right away.
        self.timer = Qt.QTimer(self)
        self.timer.timeout.connect(self.render_next_scene)  # noqa
        if loop:
            self.toggle_looping()

    def render_next_scene(self, resetcam: bool = False) -> None:
        scene_dict = self.get_next_scene_dict()
        if scene_dict is not None:
            self.render_(scene_dict, resetcam=resetcam)
        elif self.timer.isActive():
            self.timer.stop()

    def _on_key(self, event_dict) -> None:
        super(MainWindowGenerator, self)._on_key(event_dict)
        key_pressed = event_dict["keyPressed"]
        if key_pressed == "s":
            self.toggle_looping()
        elif key_pressed == "n":
            self.render_next_scene()

    @staticmethod
    def print_usage():
        MainWindow.print_usage()
        print("\ts: play / pause",
              "\n\tn: next")

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


def display_generated_custom(
    main_window_class: Type[MainWindowGenerator],
    scene_iterator: Iterable[SceneDictType],
    horizontal: bool = True,
    loop: bool = False,
    fps: float = 30.0,
    video_directory: Optional[Union[pathlib.Path, str]] = None,
    show_labels: bool = False,
    use_scene_cam: bool = False,
    debug: bool = False
):
    """Display scenes stored in scene iterator as PyQt app using custom main window class.

    The scene iterator yields a dictionary that describes the elements of the scene, one dictionary per frame.
    See SceneDictType for currently supported object types.

    Args:
        main_window_class: custom main window class.
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
    window = main_window_class(
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


def display_generated(
    scene_iterator: Iterable[SceneDictType],
    horizontal: bool = True,
    loop: bool = False,
    fps: float = 30.0,
    video_directory: Optional[Union[pathlib.Path, str]] = None,
    show_labels: bool = False,
    use_scene_cam: bool = False,
    debug: bool = False
):
    """Display scenes stored in scene iterator as PyQt app.

    The scene iterator yields a dictionary that describes the elements of the scene, one dictionary per frame.
    See SceneDictType for currently supported object types.

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
    return display_generated_custom(
        main_window_class=MainWindowGenerator,
        scene_iterator=scene_iterator,
        horizontal=horizontal,
        loop=loop,
        fps=fps,
        video_directory=video_directory,
        show_labels=show_labels,
        debug=debug,
        use_scene_cam=use_scene_cam
    )


def display_scenes(*args, **kwargs):
    print("\033[93m" + f"display_scenes() is out-dated and will be removed in the next version, "
                       f"use display_generated() instead" + "\033[0m")
    return display_generated(*args, **kwargs)
