from typing import Union

import imageio as iio
import numpy as np

from .base_io import BaseIO

class VolumeIO(BaseIO):
    Itype = int
    Otype = int

    @staticmethod
    def input(path: str, **kwargs) -> Itype:
        raise NotImplementedError

    @staticmethod
    def output(path: str, data: Otype, **kwargs) -> None:
        raise NotImplementedError
