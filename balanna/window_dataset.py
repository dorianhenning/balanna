import copy
import numpy as np
import pathlib
import pickle as pkl

from matplotlib.axes import Axes
from PyQt5 import Qt
from typing import List, Optional, Union, Type

from .window_base import MainWindow, SceneDictType
from .utils.types import contains_scene


__all__ = ['display_dataset', 'display_dataset_custom', 'MainWindowDataset']


class MainWindowDataset(MainWindow):

    def __init__(
        self,
        scenes: List[SceneDictType],
        fps: float,
        video_directory: Optional[Union[pathlib.Path, str]] = None,
        horizontal: bool = True,
        loop: bool = True,
        show_labels: bool = False,
        use_scene_cam: bool = False,
        store_directory: Optional[Union[pathlib.Path, str]] = None,
        debug: bool = False,
        parent: Qt.QWidget = None
    ):
        self.scenes = scenes
        self.fps = fps
        if len(scenes) == 0:
            return
        self.__frame_idx = 0
        scene_dict = self.scenes[0]

        image_keys = [key for key, value in scene_dict.items() if isinstance(value, np.ndarray)]
        scene_keys = [key for key, value in scene_dict.items() if contains_scene(value)]
        figure_keys = [key for key, value in scene_dict.items() if isinstance(value, Axes)]
        super(MainWindowDataset, self).__init__(
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

        # If store_directory is given, save the scene list to the directory.
        if store_directory is not None:
            print("\033[36m" + f"Storing scene in directory {store_directory}" + "\033[0m")
            store_directory = pathlib.Path(store_directory)
            if store_directory.exists():
                print("\033[33m" + f"Directory {store_directory} already exists, overwriting..." + "\033[0m")
            store_directory.mkdir(parents=True, exist_ok=True)
            for k, scene in enumerate(scenes):
                pkl_file = store_directory / f"{k:05d}.pkl"
                with open(pkl_file, 'wb') as f:
                    pkl.dump(scene, f, protocol=pkl.HIGHEST_PROTOCOL)
            print("\033[36m" + f"... done" + "\033[0m")

        # Setup looping timer and connect it to render a new scene at every timer callback.
        # If loop is true, start looping right away.
        self.timer = Qt.QTimer(self)
        self.timer.timeout.connect(self.render_loop)  # noqa
        if loop:
            self.toggle_looping()

    def render_loop(self):
        self.render_current_index()
        self.__frame_idx += 1  # TODO: run forward or backward

    def render_current_index(self, resetcam: bool = False) -> None:
        if 0 <= self.__frame_idx < len(self.scenes):
            scene_dict = self.scenes[self.__frame_idx]

            # The same matplotlib.Axes cannot be used multiple times, as it is assigned to a figure during
            # plotting. Therefore, we need to create a copy of the scene dictionary to be able to reuse it
            # when plotting the scene again later.
            if any(isinstance(value, Axes) for value in scene_dict.values()):
                scene_dict_safe = dict()
                for key, value in scene_dict.items():
                    if key in self.figure_key_dict.keys():
                        value = copy.deepcopy(value)
                    scene_dict_safe[key] = value
            else:
                scene_dict_safe = scene_dict

            self.render_(scene_dict_safe, resetcam=resetcam)
        else:
            # Reset frame index to a valid index.
            self.__frame_idx = max(min(self.__frame_idx, len(self.scenes) - 1), 0)
            # Stop the timer, if it is running.
            if self.timer.isActive():
                self.timer.stop()

    def _on_key(self, event_dict) -> None:
        super(MainWindowDataset, self)._on_key(event_dict)
        key_pressed = event_dict["keyPressed"]
        if key_pressed == "s":
            self.toggle_looping()
        elif key_pressed == "n" and not self.timer.isActive():
            self.__frame_idx += 1
            self.render_current_index()
        elif key_pressed == "b" and not self.timer.isActive():
            self.__frame_idx -= 1
            self.render_current_index()

    @staticmethod
    def print_usage():
        MainWindow.print_usage()
        print("\ts: play / pause",
              "\n\tn: next",
              "\n\tb: previous")

    def toggle_looping(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(int(1 / self.fps * 1000))  # in milliseconds


def display_dataset_custom(
    main_window_class: Type[MainWindowDataset],
    scenes: List[SceneDictType],
    horizontal: bool = True,
    loop: bool = False,
    fps: float = 30.0,
    video_directory: Optional[Union[pathlib.Path, str]] = None,
    show_labels: bool = False,
    use_scene_cam: bool = False,
    store_directory: Optional[Union[pathlib.Path, str]] = None,
    debug: bool = False
):
    """Display scenes stored in scene iterator as PyQt app with a custom main window dataset class.

    The scene iterator yields a dictionary that describes the elements of the scene, one dictionary per frame.
    See SceneDictType for currently supported object types.

    Args:
        main_window_class
        scenes: sorted list of scene dictionaries.
        horizontal: window orientation, horizontal or vertical stacking.
        loop: start looping from beginning.
        fps: frames (i.e. scenes) per second for looping.
        video_directory: directory for storing screenshots.
        show_labels: display the scene dict keys.
        use_scene_cam: use camera transform from trimesh scenes.
        store_directory: path to directory storing scene pickle files.
        debug: printing debug information.
    """
    app = Qt.QApplication([])
    window = main_window_class(
        scenes=scenes,
        fps=fps,
        video_directory=video_directory,
        horizontal=horizontal,
        loop=loop,
        show_labels=show_labels,
        debug=debug,
        use_scene_cam=use_scene_cam,
        store_directory=store_directory
    )
    app.aboutToQuit.connect(window.on_close)
    app.exec_()


def display_dataset(
    scenes: List[SceneDictType],
    horizontal: bool = True,
    loop: bool = False,
    fps: float = 30.0,
    video_directory: Optional[Union[pathlib.Path, str]] = None,
    show_labels: bool = False,
    use_scene_cam: bool = False,
    store_directory: Optional[Union[pathlib.Path, str]] = None,
    debug: bool = False
):
    """Display scenes stored in scene iterator as PyQt app.

    The scene iterator yields a dictionary that describes the elements of the scene, one dictionary per frame.
    See SceneDictType for currently supported object types.

    Args:
        scenes: sorted list of scene dictionaries.
        horizontal: window orientation, horizontal or vertical stacking.
        loop: start looping from beginning.
        fps: frames (i.e. scenes) per second for looping.
        video_directory: directory for storing screenshots.
        show_labels: display the scene dict keys.
        use_scene_cam: use camera transform from trimesh scenes.
        store_directory: path to directory storing scene pickle files.
        debug: printing debug information.
    """
    return display_dataset_custom(
        main_window_class=MainWindowDataset,
        scenes=scenes,
        horizontal=horizontal,
        loop=loop,
        fps=fps,
        video_directory=video_directory,
        show_labels=show_labels,
        use_scene_cam=use_scene_cam,
        store_directory=store_directory,
        debug=debug
    )
