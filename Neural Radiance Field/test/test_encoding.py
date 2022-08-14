import os
import sys
import unittest

sys.path.append(os.curdir)
from ioutil import JsonIO
from model.encoding import make_encoder

class TestCase(unittest.TestCase):
    def setUp(self):
        self.root = r"D:\Mywork\Computer Graphics\Neural Radiance Field\config\hotdog.json"
        self.config = JsonIO.input(self.root)

    def test(self):
        encoder = make_encoder(self.config["encoding"])
        print(encoder.in_dim)
        print(encoder.out_dim)

if __name__ == "__main__":
    unittest.main(warnings="ignore")
