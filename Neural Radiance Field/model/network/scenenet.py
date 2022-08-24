import torch
import torch.nn as nn

from ..encoding import make_encoder

class SceneNet(nn.Module):
    """NeRF, MLP represented scene."""
    def __init__(self, config: dict):
        super(SceneNet, self).__init__()

        mlp_width = config["mlp_width"]

        # positional encoding for input position
        self.pos_encoder = make_encoder(config["encoding"])
        pos_embed_width = self.pos_encoder.out_dim

        # positional encoding for input direction
        self.dir_encoder = make_encoder(config["dir_encoding"])
        dir_embed_width = self.dir_encoder.out_dim

        # stage 1 of NeRF MLP
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
