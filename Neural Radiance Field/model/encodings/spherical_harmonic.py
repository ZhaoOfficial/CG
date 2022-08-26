import torch

from .encoding import Encoder

from ..gen_shs import gen_shs

class SphericalHarmonicEncoder(Encoder):
    """
    Using a set of spherical harmonics as encoding functions. 
    """

    def __init__(self, config: dict) -> None:
        super(SphericalHarmonicEncoder, self).__init__(config)

        self.in_dim = config["in_dim"]
        assert self.in_dim == 3, "spherical harmonics only support to encode 3-D vectors."
        self.num_degrees = config["num_degrees"]
        self.out_dim = self.num_degrees ** 2

        self.enc_funcs = gen_shs(self.num_degrees)

    def forward(self, tensor_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        # tensor_in.size() = (N, self.in_dim)
        # tensor_out.size() = (N, self.out_dim)
        tensor_out = torch.zeros(*tensor_in.shape[:-1], self.out_dim)
        # x, y, z.size() = (N, 1)
        x, y, z = torch.split(tensor_in, 1, dim=-1)
        return tensor_out
