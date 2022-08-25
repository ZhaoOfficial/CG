from functools import partial
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import string_case_insensitive

class Exponetial(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(input)

def make_activation_module(config: dict) -> Union[nn.Identity, Exponetial, nn.ReLU, nn.Sigmoid, nn.Softplus]:
    """Return a activation module for a given name."""

    name = config["name"]
    # Default args for the activation module
    defaults = config.copy()
    defaults.pop("name")

    if string_case_insensitive(name, "None"):
        return nn.Identity()
    elif string_case_insensitive(name, "Exponetial"):
        return Exponetial()
    elif string_case_insensitive(name, "ReLU"):
        return nn.ReLU(**defaults)
    elif string_case_insensitive(name, "Sigmoid"):
        return nn.Sigmoid()
    elif string_case_insensitive(name, "Softplus"):
        return nn.Softplus(**defaults)
    else:
        raise KeyError("Wrong activation module name: got `{}`.".format(name))

def make_activation_function(config: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return a activation function for a given name.
    Here the default args of some activation functions have been binded.
    One only need to use it as an unary function.
    """

    name = config["name"]
    # Default args for the activation function
    defaults = config.copy()
    defaults.pop("name")

    if string_case_insensitive(name, "None"):
        return lambda x: x
    elif string_case_insensitive(name, "Exponetial"):
        return torch.exp
    elif string_case_insensitive(name, "ReLU"):
        return partial(F.relu, **defaults)
    elif string_case_insensitive(name, "Sigmoid"):
        return torch.sigmoid
    elif string_case_insensitive(name, "Softplus"):
        return partial(F.softplus, **defaults)
    else:
        raise KeyError("Wrong activation module name: got `{}`.".format(name))
