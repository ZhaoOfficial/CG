from configparser import ConfigParser
from typing import Union
import math

import torch
import torch.nn as nn

from utils import logging

logger = logging.Logger("model/__init__")

def makeModel(config: ConfigParser) -> nn.Module:
    pass

def makeLossFunction(config: ConfigParser) -> Union[nn.SmoothL1Loss, nn.MSELoss]:
    loss_fn_name = config.get("model", "LOSS_FN")
    if loss_fn_name == "L1":
        return nn.SmoothL1Loss()
    elif loss_fn_name == "L2":
        return nn.MSELoss()
    else:
        raise ValueError(
            "Wrong loss function name: got `{}`. Valid names are `L1` and `L2`.".format(loss_fn_name)
        )

def makeOptimizer(config: ConfigParser, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_name = config.get("model", "optimizer")
    lr = config.getfloat("model", "learning_rate")
    weight_decay = config.getfloat("model", "weight_decay")
    beta1 = config.getfloat("model", "beta1")
    beta2 = config.getfloat("model", "beta2")
    eps = config.getfloat("model", "eps")

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            "Wrong optimizer name: got `{}`. Valid name is `Adam`.".format(optimizer_name)
        )

    return optimizer

def makeScheduer(config: ConfigParser, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    
    warmup_iters = config.getint("model", "warmup_iters")
    start_iters = config.getint("model", "start_iters")
    end_iters = config.getint("model", "end_iters")
    lr_scale = config.getfloat("model", "lr_scale")

    def scheduler(epoch):
        """`epoch` start from 1."""
        if epoch <= warmup_iters:
            return epoch / warmup_iters

        if epoch >= start_iters:
            return (1.0 - lr_scale) * math.exp(-(epoch - start_iters) / (end_iters - start_iters)) + lr_scale

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=[scheduler] * len(optimizer.param_groups)
    )
