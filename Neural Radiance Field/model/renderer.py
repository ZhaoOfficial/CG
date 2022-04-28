from configparser import ConfigParser

import torch
import torch.nn as nn

from .scenenet import SceneNet
from utils import logging

logger = logging.Logger("model/renderer")

class Renderer(nn.Module):
    def __init__(self, config: ConfigParser):
        super(Renderer, self).__init__()

        self.num_coarse_sample = config.getint("model", "num_coarse_sample")
        self.num_fine_sample = config.getint("model", "num_fine_sample")
        self.sample_method = config.get("model", "sample_method")

        #* NeRF, the coarse and fine scene
        self.scene_net = SceneNet(config)
        if config.getbool("model", "use_same_scene_net"):
            self.scene_net_fine = self.scene_net
        else:
            self.scene_net_fine = SceneNet(config)

        #* Volumn rendering
        self.volumn_renderer = VolumeRenderer(config)

    def forward(self,):
        pass