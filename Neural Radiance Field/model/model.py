import torch
import torch.nn as nn

class Model(object):
    def __init__(self, config: dict) -> None:
        self.training_set = None
        self.test_set = None
        self.scene_net = None

    def train(self,) -> None:
        pass

    def eval(self,) -> None:
        pass
