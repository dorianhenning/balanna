import io

import PIL.Image
import pyglet
import numpy as np
import trimesh


def to_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


def from_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(180), [1, 0, 0]
    )


def numpy_to_image(arr):
    with io.BytesIO() as f:
        PIL.Image.fromarray(arr).save(f, format="PNG")
        return pyglet.image.load(filename=None, file=f)
