import torch

from utils import logger

logger = logger.Logger("model/pos_enc")

class PositionalEncoder(object):
    def __init__(
        self, L: int, input_dims: int = 3, include_input: bool = True, log_sampling: bool = True
    ):
        self.L = L

        # self.frequency.size() = (L, )
        if log_sampling:
            # equidistant in logarithm space
            self.frequency = 2.0 ** torch.linspace(0.0, float(L - 1), L) * torch.pi
        else:
            # equidistant in linear space
            self.frequency = torch.linspace(0.0, 2.0 ** float(L - 1), L) * torch.pi

        self.funcs = []
        if include_input:
            self.funcs += [lambda x: x]

        basic_funcs = [torch.sin, torch.cos]
        #? sin(2 ** 0 * x), cos(2 ** 0 * x), ..., sin(2 ** (L - 1) * x), cos(2 ** (L - 1) * x)
        #? sin(2 ** 0 * pi * x), cos(2 ** 0 * pi * x), ..., sin(2 ** (L - 1) * pi * x), cos(2 ** (L - 1) * pi * x)
        self.funcs += [lambda x, func = func, freq = freq: func(x * freq) for func in basic_funcs for freq in self.frequency]
        self.output_dim = len(self.funcs) * input_dims

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # size() = (N, len(self.funcs)) -> (N, len(self.funcs) * 3)
        return torch.cat([func(x) for func in self.funcs], dim = -1)

    @property
    def dimension(self) -> int:
        return self.output_dim
