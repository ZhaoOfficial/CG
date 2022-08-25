import torch
import torch.nn as nn

from ..encoding import make_encoder

class NerfNetwork(nn.Module):
    """NeRF, MLP represented scene."""
    def __init__(self, config: dict):
        super(NerfNetwork, self).__init__()

        #* Encoder for input position
        self.pos_encoder = make_encoder(config["encoding"]["pos_encoding"])
        #* Encoder for input direction
        self.dir_encoder = make_encoder(config["encoding"]["dir_encoding"])

        pos_embed_width = self.pos_encoder.out_dim

        #* Stage 1 of NeRF MLP
        stage_1_config = config["network"]["stage_1_network"]
        self.stage1 = nn.Sequential(
            nn.Linear(pos_embed_width, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(inplace=True),
        )

        # stage 2 of NeRF MLP, starting from a skip connection
        self.stage2 = nn.Sequential(
            nn.Linear(pos_embed_width + mlp_width, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(inplace=True),
        )

        # net for predicting density
        self.density_net = nn.Linear(mlp_width, 1)

        # net for connecting stage2 and rgb
        self.feature_net = nn.Linear(mlp_width, mlp_width)

        dir_embed_width = self.dir_encoder.out_dim

        # net for predicting rgb
        self.rgb_net = nn.Sequential(
            nn.Linear(dir_embed_width + mlp_width, mlp_width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:

        return
        return {
            "density": density,
            "rgb": rgb
        }
