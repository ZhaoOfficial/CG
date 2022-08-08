import os
import os.path as osp
import sys
import unittest

sys.path.append(os.curdir)
from ioutil import JsonIO

class Test(unittest.TestCase):
    def setUp(self):
        self.root = r"D:\Mywork\Computer Graphics\Cuda\image\heat\config"

    def testInput(self):
        data = JsonIO.input(osp.join(self.root, "hotdog.json"))
        print(data)

    def testOutput(self):
        data = [{'A': 1}, {'B': 2}]
        JsonIO.output(osp.join(self.root, "test.json"), data)

if __name__ == "__main__":
    unittest.main()
