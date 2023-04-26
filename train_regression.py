#!/usr/bin/env python3

"""
Script to train the agent through reinforcement learning.
"""

import os
import logging
import csv
import json
import datetime
import torch
import numpy as np
import sys
import time
from pathlib import Path

from arguments import ArgumentParser
from models import RModel

import utils
from regression import Regression, StockLoader

root = Path(__file__).absolute()
log_dir = os.path.join(root.parent, 'logs')
os.environ['STOCK_STORAGE'] = log_dir

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main():

    # Parse arguments
    parser = ArgumentParser()

    args = parser.parse_args()

    # set seed
    utils.seed(args.seed)

    train_dl, val_dl, test_dl = StockLoader(args.stock_filepaths, args.obs_columns, args.batch_size, args.w)

    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    default_model_name = "Regression_" + args.flag + "_" + suffix
    
    args.model = default_model_name

    rmodel = RModel(len(args.obs_columns), memory_dim=args.memory_dim)
    utils.save_model(rmodel, args.model)

    rmodel = rmodel.to(device)

    algo = Regression(rmodel, train_dl, val_dl, test_dl, device=device, lr=args.lr)
    
    utils.seed(args.seed)

    utils.save_obj(args, args.model, 'args.pkl')

    history = algo.train(num_epoch=100, val_turn=10)
    
    utils.save_obj(history, args.model, 'history.pd')

        



if __name__ == '__main__':
    main()
