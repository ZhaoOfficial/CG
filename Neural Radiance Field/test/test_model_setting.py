import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.append(os.curdir)
from model.settings import make_activation_module
from model.settings import make_activation_function
from model.settings.activations import Exponential

class TestModule(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(5, 5, requires_grad=True)
        # Reference answer
        self.ref_relu = self.input.clone()
        self.ref_relu[self.ref_relu < 0] = 0
        self.ref_grad_relu = self.ref_relu.clone()
        self.ref_grad_relu[self.ref_grad_relu > 0] = 1

        self.ref_sigmoid = 1 / (1 + torch.exp(-self.input))
        self.ref_grad_sigmoid = self.ref_sigmoid * (1 - self.ref_sigmoid)

        self.ref_softplus_1 = torch.log(1 + torch.exp(self.input))
        self.ref_grad_softplus_1 = self.ref_sigmoid

        self.ref_softplus_2 = torch.where(self.input < 10 / 10, torch.log(1 + torch.exp(10 * self.input)) / 10, self.input)
        self.ref_grad_softplus_2 = torch.where(self.input < 10 / 10, 1 / (1 + torch.exp(-10 * self.input)), 1)

    def test_none(self):
        act_config = {
            "name": "None"
        }
        act = make_activation_module(act_config)
        self.assertIsInstance(act, nn.Identity)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.input, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(torch.ones_like(self.input), self.input.grad))

    def test_Exponential(self):
        act_config = {
            "name": "Exponential"
        }
        act = make_activation_module(act_config)
        self.assertIsInstance(act, Exponential)
        output = act(self.input)
        self.assertTrue(torch.allclose(torch.exp(self.input), output))
        output.sum().backward()
        self.assertTrue(torch.allclose(torch.exp(self.input), self.input.grad))

    def test_relu_1(self):
        act_config = {
            "name": "ReLU"
        }
        act = make_activation_module(act_config)
        self.assertIsInstance(act, nn.ReLU)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_relu, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_relu, self.input.grad))

    def test_relu_2(self):
        act_config = {
            "name": "ReLU",
            "inplace": False
        }
        act = make_activation_module(act_config)
        self.assertIsInstance(act, nn.ReLU)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_relu, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_relu, self.input.grad))

    def test_sigmoid(self):
        act_config = {
            "name": "Sigmoid"
        }
        act = make_activation_module(act_config)
        self.assertIsInstance(act, nn.Sigmoid)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_sigmoid, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_sigmoid, self.input.grad))

    def test_softplus_1(self):
        act_config = {
            "name": "Softplus"
        }
        act = make_activation_module(act_config)
        self.assertIsInstance(act, nn.Softplus)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_softplus_1, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_softplus_1, self.input.grad))

    def test_softplus_2(self):
        act_config = {
            "name": "Softplus",
            "beta": 10,
            "threshold": 10
        }
        act = make_activation_module(act_config)
        self.assertIsInstance(act, nn.Softplus)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_softplus_2, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_softplus_2, self.input.grad))

class TestFunction(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(5, 5, requires_grad=True)
        # Reference answer
        self.ref_relu = self.input.clone()
        self.ref_relu[self.ref_relu < 0] = 0
        self.ref_grad_relu = self.ref_relu.clone()
        self.ref_grad_relu[self.ref_grad_relu > 0] = 1

        self.ref_sigmoid = 1 / (1 + torch.exp(-self.input))
        self.ref_grad_sigmoid = self.ref_sigmoid * (1 - self.ref_sigmoid)

        self.ref_softplus_1 = torch.log(1 + torch.exp(self.input))
        self.ref_grad_softplus_1 = self.ref_sigmoid

        self.ref_softplus_2 = torch.where(self.input < 10 / 10, torch.log(1 + torch.exp(10 * self.input)) / 10, self.input)
        self.ref_grad_softplus_2 = torch.where(self.input < 10 / 10, 1 / (1 + torch.exp(-10 * self.input)), 1)

    def test_none(self):
        act_config = {
            "name": "None"
        }
        act = make_activation_function(act_config)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.input, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(torch.ones_like(self.input), self.input.grad))

    def test_Exponential(self):
        act_config = {
            "name": "Exponential"
        }
        act = make_activation_function(act_config)
        output = act(self.input)
        self.assertTrue(torch.allclose(torch.exp(self.input), output))
        output.sum().backward()
        self.assertTrue(torch.allclose(torch.exp(self.input), self.input.grad))

    def test_relu_1(self):
        act_config = {
            "name": "ReLU"
        }
        act = make_activation_function(act_config)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_relu, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_relu, self.input.grad))

    def test_relu_2(self):
        act_config = {
            "name": "ReLU",
            "inplace": False
        }
        act = make_activation_function(act_config)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_relu, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_relu, self.input.grad))

    def test_sigmoid(self):
        act_config = {
            "name": "Sigmoid"
        }
        act = make_activation_function(act_config)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_sigmoid, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_sigmoid, self.input.grad))

    def test_softplus_1(self):
        act_config = {
            "name": "Softplus"
        }
        act = make_activation_function(act_config)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_softplus_1, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_softplus_1, self.input.grad))

    def test_softplus_2(self):
        act_config = {
            "name": "Softplus",
            "beta": 10,
            "threshold": 10
        }
        act = make_activation_function(act_config)
        output = act(self.input)
        self.assertTrue(torch.allclose(self.ref_softplus_2, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(self.ref_grad_softplus_2, self.input.grad))

if __name__ == "__main__":
    unittest.main()
