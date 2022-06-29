import glob
import os

import imageio as iio
from PIL import Image

def genVideo(in_path: str, out_path: str):
    paths = glob.glob(in_path)
    # sort by number
    paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    images = [Image.open(path) for path in paths]
    iio.mimwrite(out_path, images, fps=10, quality=10)

if __name__ == '__main__':
    genVideo(r'image\ripple\*.png', r'image\ripple.mp4')
