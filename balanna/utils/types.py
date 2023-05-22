import trimesh
import vedo


def contains_scene(x):
    return isinstance(x, trimesh.Scene) or isinstance(x, vedo.Volume)
