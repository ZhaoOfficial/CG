import torch
import torch.nn as nn

class Encoder(nn.Module):
    """The base class for all encoder."""
    def __init__(self, config: dict) -> None:
        self.in_dim = None
        self.out_dim = None

    def forward(self, tensor_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
