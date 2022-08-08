import os
import os.path as osp
import sys
import unittest

import numpy as np

sys.path.append(os.curdir)
from ioutil import ImageIO, VideoIO

class Test(unittest.TestCase):
    def setUp(self):
        self.root = r"D:\Mywork\Computer Graphics\Cuda\image"

    def testInput(self):
        data = VideoIO.input(osp.join(self.root, "heat.mp4"))
        self.assertEqual(len(data), 400)
        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], np.ndarray)

    def testOutput(self):
        data = ImageIO.inputDirectory(osp.join(self.root, "heat"), to_ndarray = True, key = lambda x: int(osp.splitext(osp.basename(x))[0]))
        VideoIO.output(osp.join(self.root, "new_heat.mp4"), data)

if __name__ == "__main__":
    unittest.main(warnings="ignore")
