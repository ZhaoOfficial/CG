import json
from typing import Any

from .base_io import BaseIO

class JsonIO(BaseIO):
    """An encapsulation of reading and writing JSON files."""

    Itype = Any
    Otype = Any

    @staticmethod
    def input(path: str, **kwargs) -> Itype:
        with open(path, 'r') as f:
            data = json.load(f, **kwargs)
        return data

    @staticmethod
    def output(path: str, data: Otype, **kwargs) -> None:
        """If the number of indentation is not specified, default to 2."""

        if kwargs.get('indent') is None:
            kwargs['indent'] = 2
        with open(path, 'w') as f:
            json.dump(data, f, **kwargs)
