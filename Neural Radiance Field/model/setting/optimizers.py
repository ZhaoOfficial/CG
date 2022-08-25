import torch
import torch.nn as nn

def make_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
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
        raise KeyError("Wrong optimizer name: got `{}`.".format(optimizer_name))

    return optimizer
