from configparser import ConfigParser

from torch.utils import data

from utils import logging
from .ray_dataset import SyntheticDataset

logger = logging.Logger("dataset/build")

def makeTrainLoader(config: ConfigParser) -> data.DataLoader:

    batch_size = config.getint("dataset", "batch_size")
    dataset_name = config.get("dataset", "dataset_name")
    if dataset_name == "synthetic":
        dataset = SyntheticDataset(config)
    else:
        raise ValueError(
            "Wrong dataset name: got `{}`. Valid names are `synthetic` and `real`.".format(dataset_name)
        )

    num_workers = config.getint("dataset", "num_workers")
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
    )

    return data_loader
