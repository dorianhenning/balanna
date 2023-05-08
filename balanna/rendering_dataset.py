import numpy as np
import trimesh
import vedo
import videoio

from typing import List

from .utils.vedo_utils import trimesh_scene_2_vedo
from .window_base import SceneDictType


__all__ = ["render_dataset"]


def render_dataset(
    scenes: List[SceneDictType],
    show_labels: bool = False,
    use_scene_cam: bool = False,
    debug: bool = False
):
    if len(scenes) == 0:
        return np.zeros((0, 640, 480, 3), dtype=np.uint8)

    num_scenes = [len(scene_dict) for scene_dict in scenes]
    assert num_scenes.count(num_scenes[0]) == len(num_scenes)  # all scenes have same length
    num_scenes = num_scenes[0]  # length of scene dict

    plotter = vedo.Plotter(num_scenes)
    images = []
    for scene_dict in scenes:
        plotter.clear()
        for i, (key, scene) in enumerate(scene_dict.items()):
            if not isinstance(scene, trimesh.Scene):
                print("\033[93m" + f"Only trimesh.Scene type is supported for rendering, skipping {key}..." + "\033[0m")
            meshes_vedo, camera_dict = trimesh_scene_2_vedo(
                scene=scene,
                label=key if show_labels else None,
                use_scene_cam=use_scene_cam
            )
            plotter.at(i).show(meshes_vedo, camera=camera_dict)
        images_i = plotter.screenshot(asarray=True)
        images.append(images_i)

    plotter.close()
    return np.stack(images)
