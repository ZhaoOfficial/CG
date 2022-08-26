import torch
import torch.nn as nn

from utils.utils import string_case_insensitive

def make_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Return an optimizer function for a given name."""

    name = config["name"]
    # Default args for the activation module
    defaults = config.copy()
    defaults.pop("name")

    if string_case_insensitive(name, "Adam"):
        return torch.optim.Adam(params=model.parameters(), **defaults)
    elif string_case_insensitive(name, "SGD"):
        return torch.optim.SGD(params=model.parameters(), **defaults)
    else:
        raise KeyError("Wrong optimizer name: got `{}`.".format(name))
