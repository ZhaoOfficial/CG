import torch
import torch.nn as nn

from ..encodings import make_encoder
from ..settings import make_activation_module

class NeRFNetwork(nn.Module):
    """NeRF, MLP represented scene."""
    def __init__(self, config: dict):
        super(NeRFNetwork, self).__init__()

        #* Encoder for input position
        self.pos_encoder = make_encoder(config["encoding"]["pos_encoding"])
        #* Encoder for input direction
        self.dir_encoder = make_encoder(config["encoding"]["dir_encoding"])

        #* Stage 1 of NeRF MLP
        self.stage1 = self.make_networks(
            config["network"]["stage_1_network"],
            in_dim=self.pos_encoder.out_dim
        )

        #* Stage 2 of NeRF MLP, starting from a skip connection
        self.stage2 = self.make_networks(
            config["network"]["stage_2_network"],
            in_dim=self.pos_encoder.out_dim + config["network"]["stage_2_network"]["mlp_width"]
        )

        #* Network for predicting density
        self.density_net = self.make_networks(
            config["network"]["density_network"],
            out_dim=1
        )

        #* Network for connecting stage2 and rgb
        self.feature_net = self.make_networks(
            config["network"]["feature_network"]
        )
    
        #* Network for predicting rgb, no activation function after the last layer
        self.rgb_net = self.make_networks(
            config["network"]["rgb_network"],
            in_dim=config["network"]["stage_2_network"]["mlp_width"],
            out_dim=3
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:

        rgb = torch.sigmoid()
        return

    def make_networks(self, config: dict, *, in_dim: int = None, out_dim: int = None) -> nn.Sequential:
        """Build the network with respect to the configuration"""
        network = []
        mlp_width: int = config["mlp_width"]
        num_layers: int = config["num_layers"]
        activation: dict = config["activation"]
        # with_first_activation: bool = config["with_first_activation"]
        with_last_activation: bool = config["with_last_activation"]

        if in_dim is None:
            in_dim = mlp_width
        if out_dim is None:
            out_dim = mlp_width

        if num_layers == 1:
            # if with_first_activation:
            #     network.append(make_activation_module(activation))
            network.append(nn.Linear(in_dim, out_dim))
            if with_last_activation:
                network.append(make_activation_module(activation))
        else:
            for i in range(num_layers):
                if i == 0:
                    # if with_first_activation:
                    #     network.append(make_activation_module(activation))
                    network.append(nn.Linear(in_dim, mlp_width))
                    network.append(make_activation_module(activation))
                elif i == num_layers - 1:
                    network.append(nn.Linear(mlp_width, out_dim))
                    if with_last_activation:
                        network.append(make_activation_module(activation))
                else:
                    network.append(nn.Linear(mlp_width, mlp_width))
                    network.append(make_activation_module(activation))

        return nn.Sequential(*network)
