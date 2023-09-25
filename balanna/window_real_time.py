import pathlib
import pickle as pkl

from functools import partial
from PyQt5 import Qt
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from typing import List, Optional, Union

from .window_base import MainWindow, SceneDictType


__all__ = ['display_real_time', 'RealTimeNode']


class RealTimeNode(QObject):
    """Template class for real time rendering.

    Examples:
        class DemoNode(RealTimeNode):

            def callback(self):
                scene = trimesh.Scene()
                box_mesh = trimesh.creation.box(bounds=[(-1, -1, 0), (1, 1, 1)])
                scene.add_geometry(box_mesh)
                self.render({"scene": scene})

                if condition:
                    self.close()
    """
    is_finished = False
    running = pyqtSignal()
    scene_dict_emitter = pyqtSignal(dict)

    def callback(self):
        raise NotImplementedError

    def run(self):
        while not self.is_finished:
            if not self.running:
                continue
            self.callback()

    def render(self, scene_dict: SceneDictType):
        self.scene_dict_emitter.emit(scene_dict)  # noqa

    def stop(self):
        self.is_finished = True


class CachingSignals(QObject):
    finished = pyqtSignal()


class CachingNode(Qt.QRunnable):

    def __init__(self, file_path: pathlib.Path, scene_dict: SceneDictType):
        super(CachingNode, self).__init__()
        self.file_path = file_path
        self.scene_dict = scene_dict
        self.signals = CachingSignals()

    @Qt.pyqtSlot()
    def run(self):
        with open(self.file_path, 'wb+') as f:
            pkl.dump(self.scene_dict, file=f)
        self.signals.finished.emit()


class MainWindowRealTime(MainWindow):

    def __init__(
        self,
        worker: QObject,
        image_keys: List[str],
        scene_keys: List[str],
        figure_keys: List[str],
        video_directory: Optional[Union[pathlib.Path, str]] = None,
        cache_directory: Optional[Union[pathlib.Path, str]] = None,
        horizontal: bool = True,
        show_labels: bool = False,
        use_scene_cam: bool = False,
        debug: bool = False,
        parent: Qt.QWidget = None
    ):
        super(MainWindowRealTime, self).__init__(
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

        self.cache_directory = None
        self.__cache_index = 0
        self.cache_thread_pool = None
        if cache_directory is not None:
            self.cache_directory = pathlib.Path(cache_directory)
            if self.cache_directory.exists():
                print("\033[93m" + "Cache directory already exists, overwriting it ..." + "\033[0m")
            self.cache_directory.mkdir(exist_ok=True, parents=True)
            self.cache_thread_pool = Qt.QThreadPool()
            if self.debug:
                caching_max_threads = self.cache_thread_pool.maxThreadCount()
                print("\033[36m" + f"Launched caching thread pool with max. {caching_max_threads} threads" + "\033[0m")

        # Set up multi-processing pipeline.
        self.thread = QThread(parent=self)
        self.worker = worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.scene_dict_emitter.connect(self.render_)
        self.thread.start()

        self.show()

    def render_(self, scene_dict: SceneDictType, resetcam: bool = False):
        super(MainWindowRealTime, self).render_(scene_dict, resetcam=resetcam)

        # If a cache directory is defined, write the scene dict to a pickle file, one per scene_dict.
        if self.cache_directory is not None:
            cache_file_name = self.cache_directory / f"{self.__cache_index:05d}.pkl"
            cache_node = CachingNode(cache_file_name, scene_dict)
            cache_node.signals.finished.connect(partial(self.__caching_complete, cache_file_name))
            self.cache_thread_pool.start(cache_node)
            self.__cache_index += 1

    def __caching_complete(self, cache_file_name: pathlib.Path):
        if self.debug:
            print("\033[36m" + f"Caching complete: {cache_file_name.as_posix()}" + "\033[0m")

    def _on_key(self, event_dict) -> None:
        super(MainWindowRealTime, self)._on_key(event_dict)
        key_pressed = event_dict["keyPressed"]
        if key_pressed == "s":
            self.worker.running = not self.worker.running

    @staticmethod
    def print_usage():
        MainWindow.print_usage()
        print("\ts: play / pause")

    def on_close(self):
        super(MainWindowRealTime, self).on_close()
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        if self.cache_thread_pool is not None:
            self.cache_thread_pool.waitForDone()


def display_real_time(
    worker: RealTimeNode,
    image_keys: Optional[List[str]] = None,
    scene_keys: Optional[List[str]] = None,
    figure_keys: List[str] = None,
    horizontal: bool = True,
    video_directory: Optional[Union[pathlib.Path, str]] = None,
    cache_directory: Optional[Union[pathlib.Path, str]] = None,
    show_labels: bool = False,
    use_scene_cam: bool = False,
    debug: bool = False
):
    """Display scenes from a real-time application, i.e. render in-coming scenes directly.

    The worker is a process that generates each new scene in a separate thread and passes it to the renderer.
    For a smooth rendering process and for simplification, the image and scene keys have to be known in advances
    and cannot be changed afterwards.

    Args:
        worker: generator of new scenes.
        image_keys: names of images in rendered dictionary.
        scene_keys: names of 3D scenes in rendered dictionary.
        figure_keys: names of matplotlib figures in rendered dictionary.
        horizontal: window orientation, horizontal or vertical stacking.
        video_directory: directory for storing screenshots.
        cache_directory: directory for caching the input scene dicts for re-rendering.
        show_labels: display the scene dict keys.
        use_scene_cam: use camera transform from trimesh scenes.
        debug: printing debug information.
    """
    if image_keys is None and scene_keys is None:
        raise ValueError("Neither image nor scene keys, provide at least one key!")
    if not hasattr(worker, 'run'):
        raise ValueError("Worker must have method run()")
    if image_keys is None:
        image_keys = []
    if scene_keys is None:
        scene_keys = []
    if figure_keys is None:
        figure_keys = []

    app = Qt.QApplication([])
    window = MainWindowRealTime(
        worker=worker,
        image_keys=image_keys,
        scene_keys=scene_keys,
        figure_keys=figure_keys,
        video_directory=video_directory,
        cache_directory=cache_directory,
        horizontal=horizontal,
        show_labels=show_labels,
        use_scene_cam=use_scene_cam,
        debug=debug
    )
    app.aboutToQuit.connect(window.on_close)
    app.exec_()
