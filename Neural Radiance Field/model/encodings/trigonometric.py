import torch

from .encoding import Encoder 

class TrigonometricEncoder(Encoder):
    """
    Also called positional encoder. Using a series of sin and cos functions as encoding functions.
    sin(2 ** 0 * pi * x), cos(2 ** 0 * pi * x), sin(2 ** 1 * pi * x), cos(2 ** 1 * pi * x) ...
    sin(2 ** 0 * pi * y), cos(2 ** 0 * pi * y), sin(2 ** 1 * pi * y), cos(2 ** 1 * pi * y) ...
    sin(2 ** 0 * pi * z), cos(2 ** 0 * pi * z), sin(2 ** 1 * pi * z), cos(2 ** 1 * pi * z) ...
    """

    def __init__(self, config: dict) -> None:
        super(TrigonometricEncoder, self).__init__(config)

        self.in_dim = config["in_dim"]
        self.num_frequencies = config["num_frequencies"]
        assert self.num_frequencies > 0, "num_frequencies should be a positive integer."
        self.out_dim = self.in_dim * self.num_frequencies * 2

        # frequency.size() = (self.num_frequencies)
        if config["log_sampling"]:
            # equidistant in logarithm space
            self.frequency = 2.0 ** torch.linspace(0.0, float(self.num_frequencies - 1), self.num_frequencies)
        else:
            # equidistant in linear space
            self.frequency = torch.linspace(0.0, 2.0 ** float(self.num_frequencies - 1), self.num_frequencies)
        self.frequency = self.frequency * torch.pi

        self.enc_funcs = []
        for freq in self.frequency:
            for func in [torch.sin, torch.cos]:
                self.enc_funcs.append(lambda x, freq=freq, func=func: func(x * freq))

    def forward(self, tensor_in: torch.Tensor) -> torch.Tensor:
        # tensor_in.size() = (N, self.in_dim)
        # tensor_out.size() = (N, self.out_dim)
        tensor_out = torch.cat([func(tensor_in) for func in self.enc_funcs], dim=-1)
        return tensor_out
