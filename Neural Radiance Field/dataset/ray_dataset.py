from configparser import ConfigParser
import os
from typing import Tuple

import torch
from torch.utils import data
import tqdm

from .frame_dataset import SyntheticImageDataset
from .ray_sampling import sampleRaySynthetic
from utils import logging

logger = logging.Logger("dataset/ray_dataset")

class SyntheticDataset(data.Dataset):
    def __init__(self, config: ConfigParser) -> None:
        super(SyntheticDataset, self).__init__()

        #* IO settings
        self.output_dir = config.get("output", "output_dir")
        self.temp_rays = config.get("dataset", "temp_rays")
        self.dataset_path = config.get("dataset", "dataset_path")

        if not os.path.exists(self.dataset_path):
            raise FileExistsError("{} does not exist".format(self.dataset_path))

        #* check if we have generated rays
        temp_ray_path = os.path.join(self.output_dir, self.temp_ray)
        if not os.path.exists(temp_ray_path):
            logger.info("No temporal rays detected, start generating new rays in {}.".format(temp_ray_path))
            os.makedirs(temp_ray_path)

        #* dataset settings
        self.image_dataset = SyntheticImageDataset(config)
        self.angles = self.image_dataset.angles
        self.camera_num = self.image_dataset.camera_num

        #* generating rays
        clear_ray = config.getboolean("dataset", "clear_ray")
        if not os.path.exists(os.path.join(temp_ray_path, "near_fars.pt")) or clear_ray:
            logger.info("Generating rays...")
            ray_list = []
            color_list = []
            near_far_list = []

            process_bar = tqdm.tqdm(range(len(self.image_dataset)), desc="Generating Image No.00 rays")
            for i in process_bar:
                # near_far.size() = (???)
                poses, intrinsic, image, mask, near_far = self.image_dataset[i]

                # rays.size() = (H * W, 6)
                # colors.size() = (H * W, 3)
                rays, colors = sampleRaySynthetic(poses, intrinsic, image)
                ray_list.append(rays)
                color_list.append(colors)
                near_far_list.append(near_far)
                
                process_bar.set_description("Generating Image No.{:02d} rays".format(i + 1))

            # self.rays.size() = (N * H * W, 6)
            self.rays = torch.cat(ray_list, dim=0)
            # self.colors.size() = (N * H * W, 6)
            self.colors = torch.cat(color_list, dim=0)
            # self.near_fars.size() = (???)
            self.near_fars = torch.cat(near_far_list, dim=0)
            logger.info("Saving rays into {}".format(temp_ray_path))
            torch.save(self.rays, os.path.join(temp_ray_path, "rays.pt"))
            torch.save(self.colors, os.path.join(temp_ray_path, "colors.pt"))
            torch.save(self.near_fars, os.path.join(temp_ray_path, "near_fars.pt"))

        else:
            logger.info("Loading rays from {}".format(temp_ray_path))
            self.rays = torch.load(os.path.join(temp_ray_path, "rays.pt"))
            self.colors = torch.load(os.path.join(temp_ray_path, "colors.pt"))
            self.near_fars = torch.load(os.path.join(temp_ray_path, "near_fars.pt"))

    def __len__(self) -> int:
        return self.rays.size(1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.rays[:, index],
            self.colors[:, index],
            self.near_fars[:, index]
        )
