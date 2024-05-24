from trimesh import Scene
from vedo import Volume


def contains_scene(x):
    return isinstance(x, Scene) or isinstance(x, Volume)
