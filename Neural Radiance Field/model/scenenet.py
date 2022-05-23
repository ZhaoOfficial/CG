from configparser import ConfigParser

import torch
import torch.nn as nn

from .pos_enc import PositionalEncoder
from utils import logger

logger = logger.Logger("model/scenenet")

class SceneNet(nn.Module):
    """NeRF, MLP represented scene."""
    def __init__(self, config: ConfigParser):
        super(SceneNet, self).__init__()

        mlp_width = config.getint("model", "mlp_width")
        include_input = config.getboolean("model", "include_input")
        self.use_dir = config.getboolean("model", "use_direction")

        # positional encoding for input position
        self.pos_pos_enc = PositionalEncoder(L = 10, include_input = include_input)
        pos_embed_width = self.pos_pos_enc.dimension

        if self.use_dir:
            # positional encoding for input direction
            self.dir_pos_enc = PositionalEncoder(L = 4, include_input = include_input)
            dir_embed_width = self.dir_pos_enc.dimension
        else:
            dir_embed_width = 0

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

    def forward(self):

        return
        return rgb, density
