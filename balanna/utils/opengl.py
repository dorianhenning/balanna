import pyglet
import numpy as np
import trimesh


def to_opengl_transform(transform: np.ndarray = None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


def to_opengl_image(arr: np.ndarray) -> pyglet.image:
    """Convert numpy array to pyglet.image.

    This function converts a numpy array to a GL image, that can be drawn by pyglet. The code for the conversion
    follows: https://stackoverflow.com/questions/9035712/numpy-array-is-shown-incorrect-with-pyglet.
    This operation can be quite slow, so to speed up, downscale the image.
    """
    from pyglet.gl import GLubyte
    bytes_per_channel = 1  # uint8 type
    if len(arr.shape) == 2:
        format_size = 1
        channel_format = "L"  # mono image
    else:
        assert arr.shape[2] == 3  # should be 3 for three color channels
        format_size = 3
        channel_format = "RGB"
    return pyglet.image.ImageData(
        width=arr.shape[1],
        height=arr.shape[0],
        format=channel_format,
        data=(GLubyte * arr.size)(*np.flipud(arr).flatten('A').astype('uint8')),  # why ever flip required ...
        pitch=arr.shape[1] * format_size * bytes_per_channel
    )
