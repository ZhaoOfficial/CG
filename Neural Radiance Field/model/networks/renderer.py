from configparser import ConfigParser

import torch
import torch.nn as nn

from sample_ray import RaySamplerBox, RaySamplerNearFar
from .nerf_network import NeRFNetwork
from utils import logger

logger = logger.Logger("model/renderer")

class Renderer(nn.Module):
    def __init__(self, config: ConfigParser):
        super(Renderer, self).__init__()

        self.num_coarse_sample = config.getint("model", "num_coarse_sample")
        self.num_fine_sample = config.getint("model", "num_fine_sample")
        self.sample_method = config.get("model", "sample_method")

        #* NeRF, representing the scene as an MLP
        self.scene_net_coarse = NeRFNetwork(config)
        if config.getbool("model", "use_same_scene_net"):
            self.scene_net_fine = self.scene_net_coarse
        else:
            self.scene_net_fine = NeRFNetwork(config)

        #* Ray sampling in coarse stage
        self.sample_method = config.get("model", "sample_method")
        if self.sample_method == "bbox":
            self.ray_sampler = RaySamplerBox(self.num_coarse_sample)
        elif self.sample_method == "near_far":
            self.ray_sampler = RaySamplerNearFar(self.num_coarse_sample)
        else:
            raise ValueError("Wrong sample method, got {}.".format(sample_method))

        #* Volumn rendering
        self.volumn_renderer = VolumeRenderer(config)

    def forward(self,):
        pass