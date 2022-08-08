import os
import os.path as osp
import sys
import unittest

from PIL import Image
import numpy as np

sys.path.append(os.curdir)
from ioutil import ImageIO

class Test(unittest.TestCase):
    def setUp(self):
        self.root = r"D:\Mywork\Computer Graphics\Cuda\image\heat"
        self.new_root = osp.join(self.root, osp.pardir)

    def testInput(self):
        # input
        image = ImageIO.input(osp.join(self.root, "0.png"))
        self.assertIsInstance(image, Image.Image)
        image = ImageIO.input(osp.join(self.root, "0.png"), to_ndarray = True)
        self.assertIsInstance(image, np.ndarray)

        # inputDirectory
        images = ImageIO.inputDirectory(osp.join(self.root))
        self.assertIsInstance(images, list)
        self.assertIsInstance(images[0], Image.Image)
        self.assertEqual(len(images), 400)
        images = ImageIO.inputDirectory(osp.join(self.root), to_ndarray = True)
        self.assertIsInstance(images, list)
        self.assertIsInstance(images[0], np.ndarray)
        self.assertEqual(len(images), 400)

    def testOutput(self):
        # output
        image = ImageIO.input(osp.join(self.root, "0.png"))
        ImageIO.output(osp.join(self.new_root, "test0.png"), image)

        # output
        images = ImageIO.inputDirectory(osp.join(self.root))
        ImageIO.outputDirectory(osp.join(self.new_root, "new_root"), images)

if __name__ == "__main__":
    unittest.main(warnings="ignore")
