# Extended from https://github.com/wkentaro/morefusion/blob/main/morefusion/extra/_trimesh/display_scenes.py
# Significant speed limitations in original version, as described in
# https://stackoverflow.com/questions/53255392/pyglet-clock-schedule-significant-function-calling-speed-limitation
import enum
import glooey
import math
import numpy as np
import pyglet
import trimesh.viewer

from balanna.utils.opengl import to_opengl_image
from typing import Optional, Tuple


class display_scenes(pyglet.window.Window):
    # TODO: re-add support for rotation around scene
    class MODE(enum.Enum):
        IDLE = 1
        NEXT = 2
        PLAYING = 3

    HEIGHT_LABEL_WIDGET = 19
    PADDING_GRID = 1

    def __init__(
        self,
        data,
        caption: Optional[str] = None,
        height: int = 480,
        width: int = 920,
        tile: Optional[Tuple[int, int]] = None
    ):
        self.data = data
        self.mode = self.MODE.IDLE

        scene_set = next(data)
        if tile is None:
            n_rows, n_cols = self._get_tile_shape(len(scene_set), hw_ratio=height / width)
        else:
            n_rows, n_cols = tile

        super(display_scenes, self).__init__(
            height=height,
            width=width,
            caption=caption
        )

        gui = glooey.Gui(self)
        grid = glooey.Grid()
        grid.set_padding(self.PADDING_GRID)
        self.widgets = {}
        trackball = None
        for i, (name, scene) in enumerate(scene_set.items()):
            vbox = glooey.VBox()

            # Add label for each widget with the label being the key of the scene dictionary.
            vbox.add(glooey.Label(text=name, color=(255,) * 3), size=0)

            if isinstance(scene, trimesh.Scene):
                self.widgets[name] = trimesh.viewer.SceneWidget(scene)
                # Share the same trackball over all 3D scene widgets.
                if trackball is None:
                    trackball = self.widgets[name].view["ball"]
                else:
                    self.widgets[name].view["ball"] = trackball

            elif isinstance(scene, np.ndarray):
                self.widgets[name] = glooey.Image(to_opengl_image(scene), responsive=True)

            else:
                raise TypeError(f"unsupported type of scene: {scene}")
            vbox.add(self.widgets[name])
            grid[i // n_cols, i % n_cols] = vbox
        gui.add(grid)

        # Launch app main loop, i.e. iteratively call callback.
        pyglet.clock.schedule(self.callback)
        pyglet.app.run()

    def callback(self, dt: float):
        if self.mode is self.MODE.IDLE:
            return

        elif self.mode in [self.MODE.PLAYING, self.MODE.NEXT]:
            try:
                scene_set = next(self.data)
                for key, widget in self.widgets.items():
                    scene = scene_set[key]
                    if isinstance(widget, trimesh.viewer.SceneWidget):
                        assert isinstance(scene, trimesh.Scene)
                        cam_tf = widget.scene.camera_transform
                        widget.clear()
                        scene.camera_transform = cam_tf  # re-store camera transform of last frame
                        widget.scene = scene
                        widget.view["ball"]._n_pose = cam_tf
                        widget._draw()
                    elif isinstance(widget, glooey.Image):
                        widget.set_image(to_opengl_image(scene))
            except StopIteration:
                self.mode = self.MODE.IDLE

            # If the mode was NEXT, set to IDLE as the next view has been rendered.
            if self.mode is self.MODE.NEXT:
                self.mode = self.MODE.IDLE

    def on_close(self):
        super(display_scenes, self).on_close()
        pyglet.clock.unschedule(self.callback)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            self.close()
        elif symbol == pyglet.window.key.H:
            print("Usage:",
                  "\n\tq: quit",
                  "\n\ts: play / pause",
                  "\n\tn: next",
                  "\n\tz: reset view",
                  "\n\tc: print camera transform")
        elif symbol == pyglet.window.key.S:
            if self.mode is self.MODE.PLAYING:
                self.mode = self.MODE.IDLE
            else:
                self.mode = self.MODE.PLAYING
        elif symbol == pyglet.window.key.N:
            self.mode = self.MODE.NEXT
        elif symbol == pyglet.window.key.Z:
            self.reset_views()

    def reset_views(self):
        for key, widget in self.widgets.items():
            if isinstance(widget, trimesh.viewer.SceneWidget):
                widget.reset_view()

    @staticmethod
    def _get_tile_shape(num: int, hw_ratio: float = 1.0):
        r_num = int(round(math.sqrt(num / hw_ratio)))  # weighted by wh_ratio
        c_num = 0
        while r_num * c_num < num:
            c_num += 1
        while (r_num - 1) * c_num >= num:
            r_num -= 1
        return r_num, c_num
