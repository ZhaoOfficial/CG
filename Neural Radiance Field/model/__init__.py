from configparser import ConfigParser
import math

import torch

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
