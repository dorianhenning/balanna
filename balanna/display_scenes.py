# Extended from https://github.com/wkentaro/morefusion/blob/main/morefusion/extra/_trimesh/display_scenes.py
# Significant speed limitations in original version, as described in
# https://stackoverflow.com/questions/53255392/pyglet-clock-schedule-significant-function-calling-speed-limitation
import enum
import math

import pyglet
import trimesh.viewer


class display_scenes(trimesh.viewer.windowed.SceneViewer):
    # TODO: re-add support for rotation around scene
    # TODO: re-add support for multiple tiles
    # TODO: re-add support for window height and width (or full-screen?)
    # TODO: re-add support for SHIFT + N
    class MODE(enum.Enum):
        IDLE = 1
        NEXT = 2
        PLAYING = 3

    def __init__(self, data, caption=None):
        self.data = data
        self.mode = self.MODE.IDLE
        scene = next(data)['point_cloud']  # TODO: remove, create tile grid
        super(display_scenes, self).__init__(
            scene=scene,
            callback=self.callback,
            caption=caption,
            start_loop=True,
        )

    def callback(self, scene: trimesh.scene):
        if self.mode is self.MODE.IDLE:
            return

        elif self.mode in [self.MODE.PLAYING, self.MODE.NEXT]:
            try:
                self.scene = next(self.data)['point_cloud']
            except StopIteration:
                print("End reached")
                self.mode = self.MODE.IDLE

            # If the mode was NEXT, set to IDLE as the next view has been rendered.
            if self.mode is self.MODE.NEXT:
                self.mode = self.MODE.IDLE

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            self.close()
        elif symbol == pyglet.window.key.H:
            print("Usage:",
                  "\n\tq: quit",
                  "\n\ts: play / pause",
                  "\n\tn: next",
                  "\n\tc: print camera transform")
        elif symbol == pyglet.window.key.S:
            if self.mode is self.MODE.PLAYING:
                self.mode = self.MODE.IDLE
            else:
                self.mode = self.MODE.PLAYING
        elif symbol == pyglet.window.key.N:
            self.mode = self.MODE.NEXT
