import glob
import os
import os.path as osp
from typing import Union, Callable

from PIL import Image
import numpy as np

from .base_io import BaseIO

class ImageIO(BaseIO):
    """
    An encapsulation of reading and writing image files using `PIL.Image` module.
    It can read from and write to a directory.
    """
    Itype = Union[np.ndarray, Image.Image, list[np.ndarray], list[Image.Image]]
    Otype = Union[np.ndarray, Image.Image, list[np.ndarray], list[Image.Image]]

    @staticmethod
    def input(path: str, *, to_ndarray = False, **kwargs) -> Itype:
        """
        Read single image file.
        `to_ndarray`: if `True`, convert to numpy array.
        """

        data = Image.open(path, **kwargs)
        if to_ndarray:
            data = np.array(data)
        return data

    @staticmethod
    def inputDirectory(path: str, *, key: Callable[[str, str], bool] = None, to_ndarray = False, **kwargs) -> Itype:
        """
        Read all the images from the `path`.
        `key`: a way to order the file path strings.
        `to_ndarray`: if `True`, convert to numpy array.
        """

        assert osp.isdir(path), "Path [{}] is not a directory".format(path)
        image_paths = glob.glob(osp.join(path, '*'))
        image_paths.sort(key = key)

        data = [ImageIO.input(image_path, to_ndarray = to_ndarray, **kwargs) for image_path in image_paths]
        if to_ndarray:
            data = list(map(np.array, data))
        return data

    @staticmethod
    def output(path: str, data: Otype, **kwargs) -> None:
        """Write single image file."""

        if isinstance(data, np.ndarray):
            data = Image.fromarray(data)
        data.save(path, **kwargs)

    @staticmethod
    def outputDirectory(path: str, data: Otype, *, name_list: list[str] = None, **kwargs) -> None:
        """
        Write all the images to the `path`.
        `name_list`: the list of names of the images to be written, default to a list of numbers and default format is `png`.
        """

        os.makedirs(path, exist_ok = True)
        assert osp.isdir(path), "Path [{}] is not a directory".format(path)

        if name_list is None:
            name_list = ["{}.png".format(i) for i in range(len(data))]
        print(len(name_list))
        for image, name in zip(data, name_list):
            ImageIO.output(osp.join(path, name), image, **kwargs)
