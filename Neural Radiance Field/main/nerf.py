import argparse
from configparser import ConfigParser
import os
import sys
import time

import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append(os.pardir)
from utils import logger
from model import makeModel, makeLossFunction, makeOptimizer, makeScheduer
from dataset import makeTrainLoader
from option.nerf_opt import parseArgument

logger = logger.Logger("main/training")

def makeConfig(args: argparse.Namespace) -> ConfigParser:
    assert os.path.exists(args.config), "Can not find configuration file with path {}".format(args.config)
    config = ConfigParser()
    config.read(args.config)
    return config

def main(
    config: ConfigParser,
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    swriter: SummaryWriter,
    resumed_epoch: int
):
    #* constant
    
    for epoch in tqdm.tqdm(dynamic_ncols=True):
        epoch_start_time = time.time()

        model.train()
        #* batch training
        batch_loss, batch_vis, batch_time = [], [], []
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()

            # TODO: here

            batch_time.append(time.time() - batch_start_time)
            batch_loss.append(loss.item())

if __name__ == "__main__":
    args = parseArgument()

    #* set pytorch gpu id and settings
    torch.cuda.set_device(args.gpu)
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)

    #* loading configuration file
    config = makeConfig(args)

    #* create ray dataset
    config.set("dataset", "clean_ray", args.clean_ray)

    #* logger and tensorboard writer
    logger.info("Configuration file path: {}".format(args.config))
    output_dir = config.get("output", "output_dir")
    swriter = SummaryWriter(log_dir=output_dir, max_queue=1)

    #* create model, optimizer, loss function, scheduler and data loader
    # TODO: from here
    model        = makeModel(config)
    loss_fn      = makeLossFunction(config)
    optimizer    = makeOptimizer(config, model)
    train_loader = makeTrainLoader(config)
    val_loader   = makeValLoader(config)
    scheduler    = makeScheduer(config)

    #* load model from checkpoint
    # TODO: from here
    if args.resume == 0:
        # specify the number of iteration
        resumed_epoch = 0
    elif args.resume == -1:
        # find the maximum number of iterations that is saved
        resumed_epoch = -1
    else:
        resumed_epoch = args.resume
        ckpt_path = os.path.join(output_dir, "checkpoint_{}.pt".format(resumed_epoch))
    
    # resume from checkpoint
    if resumed_epoch != 0:
        assert os.path.exists(ckpt_path), "Can not find the required checkpoint file: {}.".format(ckpt_path)
        logger.info("Loading checkpoint from {}, start from {} epoch.".format(ckpt_path, resumed_epoch))

        ckpt = torch.load(ckpt_path, map_location='cuda:{}'.format(args.gpu))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        logger.info("Training from scratch.")

    main(
        config, model, 
    )

