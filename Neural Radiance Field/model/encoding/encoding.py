import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config: dict):
        self.in_dim = None
        self.out_dim = None

    def forward(self, input: torch.Tensor):
        raise NotImplementedError
