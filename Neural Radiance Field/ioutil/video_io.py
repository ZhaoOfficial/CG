from typing import Union

import imageio as iio
from PIL import Image
import numpy as np

from .base_io import BaseIO

class VideoIO(BaseIO):
    """An encapsulation of reading and writing video files using `imageio.ffmpeg` module."""

    Itype = list[np.ndarray]
    Otype = Union[list[np.ndarray], list[Image.Image]]

    @staticmethod
    def input(path: str, **kwargs) -> Itype:
        reader = iio.get_reader(path, format = "ffmpeg", **kwargs)
        data = [image for image in reader]
        reader.close()
        return data

    @staticmethod
    def output(path: str, data: Otype, **kwargs) -> None:
        """If the `fps` and `quality` is not specified, default to 24 and 10 for `.mp4` file."""
        if kwargs.get("fps") is None:
            kwargs["fps"] = 24
        if kwargs.get("quality") is None:
            kwargs["quality"] = 10

        iio.mimwrite(path, data, format = "ffmpeg", **kwargs)
