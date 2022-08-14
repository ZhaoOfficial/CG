from typing import Union

import torch
import torch.nn as nn

from utils.utils import string_case_insensitive

def make_loss_function(config: dict) -> Union[nn.SmoothL1Loss, nn.MSELoss]:
    name = config["name"]
    if string_case_insensitive(name, "L1"):
        return nn.SmoothL1Loss()
    elif string_case_insensitive(name, "L2"):
        return nn.MSELoss()
    elif string_case_insensitive(name, "Huber"):
        return nn.HuberLoss()
    else:
        raise KeyError(
            "Wrong loss function name: got `{}`.".format(name)
        )
