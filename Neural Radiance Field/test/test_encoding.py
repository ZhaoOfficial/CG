import os
import sys
import unittest

import torch

sys.path.append(os.curdir)
from model.encoding import make_encoder

class TestCase(unittest.TestCase):
    def setUp(self):
        self.input = torch.Tensor([[0.1, 0.2, 0.4]])

    def test_trigonometric(self):
        encoding = {
            "name": "Trigonometric",
            "in_dim": 3,
            "num_frequencies": 10,
            "log_sampling": True
        }
        encoder = make_encoder(encoding)
        self.assertEqual(encoder.in_dim, 3)
        self.assertEqual(encoder.out_dim, encoding["in_dim"] * encoding["num_frequencies"] * 2)
        output = encoder.forward(self.input)
        self.assertEqual(output.shape[:-1], self.input.shape[:-1])
        self.assertEqual(output.shape[-1], 60)

    def test_spherical_harmonics(self):
        encoding = {
            "name": "SphericalHarmonics",
            "in_dim": 3,
            "num_degrees": 3
        }
        encoder = make_encoder(encoding)
        self.assertEqual(encoder.in_dim, 3)
        self.assertEqual(encoder.out_dim, encoding["num_degrees"] ** 2)
        # output = encoder.forward(self.input)

if __name__ == "__main__":
    unittest.main()
