from typing import Any

class BaseIO(object):
    """
    The base class for all customized IO classes.
    All the derived classes should implement the method defined in this base class.
    All the derived classes should clearly define the `Itype` and `Otype` of the class.

    `Itype`: the type of the data read from the file.
    `Otype`: the type of the data written to the file.
    """

    Itype = Any
    Otype = Any

    @staticmethod
    def input(path: str, **kwargs) -> Itype:
        raise NotImplementedError

    @staticmethod
    def output(path: str, data: Otype, **kwargs) -> None:
        raise NotImplementedError
