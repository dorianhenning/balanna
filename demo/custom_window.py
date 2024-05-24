import trimesh
import trimesh.creation

from typing import Optional

from balanna.components import show_grid
from balanna import display_generated_custom, MainWindowGenerator, SceneDictType


class MyMainWindow(MainWindowGenerator):

    def __init__(self, *args, **kwargs):
        self._toggle = False
        super(MyMainWindow, self).__init__(*args, **kwargs)

    def _on_key(self, event_dict) -> None:
        super(MyMainWindow, self)._on_key(event_dict)
        key_pressed = event_dict["keypress"]
        if key_pressed == "u":
            self._toggle = not self._toggle
            self.render_next_scene()

    def get_next_scene_dict(self) -> Optional[SceneDictType]:
        scene_dict = super(MyMainWindow, self).get_next_scene_dict()
        if self._toggle and scene_dict is not None:
            scene = scene_dict['scene']
            scene.delete_geometry('box2')
            scene_dict["scene"] = scene
        return scene_dict


def main():
    num_time_steps = 50
    for k in range(num_time_steps):
        print(f"Generated scene {k + 1}/{num_time_steps}")

        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(bounds=[(-1, -1, 0), (1, 1, 1)]), geom_name='box1')
        scene.add_geometry(trimesh.creation.box(bounds=[(-4, -4, 0), (-3, -3, 1)]), geom_name='box2')
        scene = show_grid(-10, 10, alpha=100, scene=scene)

        yield {'scene': scene}


if __name__ == '__main__':
    display_generated_custom(MyMainWindow, main(), fps=10)
