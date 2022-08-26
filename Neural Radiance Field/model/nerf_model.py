import torch
import torch.nn as nn

from networks.nerf_network import NeRFNetwork

class NeRFModel(object):
    def __init__(self, config: dict) -> None:
        self.training_set = None
        self.test_set = None
        self.network = NeRFNetwork(config)

    def train(self,) -> None:
        self.network.train()

    def render(self,) -> None:
        self.network.eval()
