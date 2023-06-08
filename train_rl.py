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
from models import ACModel
from evaluate import batch_evaluate
from agent import ModelAgent

import utils
from ppo import PPOAlgo
from stock_env import StockEnv

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

    # Generate environments
    envs = []
    for i in range(args.procs):
        env = StockEnv(stock_trend_filepaths=args.stock_filepaths, 
                       obs_column_names=args.obs_columns, w=args.w, c=args.c)
        env.seed(seed = 100 * args.seed + i)
        envs.append(env)

    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    default_model_name = "PPO_" + args.flag + "_" + suffix
    
    args.model = default_model_name

    s = utils.get_log_dir(args.model)
    utils.configure_logging(utils.get_log_dir(args.model))
    logger = logging.getLogger(__name__)

    # Define obss preprocessor
    obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space)

    # Define actor-critic model
    acmodel = None
    acmodel = ACModel(envs[0].observation_space, envs[0].action_space, 
                      memory_dim=args.memory_dim,
                      arch=args.arch, c=args.c, w=args.w)
    utils.save_model(acmodel, args.model)

    if device == 'cuda':
        acmodel.cuda()
        
    # Define actor-critic algo

    reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
    algo = PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1,
                                args.beta2,
                                args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                                reshape_reward, device=device)
    # When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
    # Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
    # the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

    utils.seed(args.seed)

    # Restore training status

    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                  'num_episodes': 0,
                  'num_frames': 0}

    # Define logger and Tensorboard writer and CSV writer

    header = (["update", "episodes", "frames", "FPS", "duration"]
              + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["success_rate"]
              + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])

    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)


    logger.info('COMMAND LINE ARGS:')
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(acmodel)

    # Train model

    total_start_time = time.time()
    best_success_rate = 0
    test_env_name = args.env

    utils.save_obj(args, args.model, 'args.pkl')

    while status['num_frames'] < args.frames:
        # Update parameters

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs
        if status['i'] % args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"]]

            format_str = (
                    f"{args.flag} ------ " + "U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                                                                                                                                                                 "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                                                                                                                                                                 "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")
            logger.info(format_str.format(*data))

            csv_writer.writerow(data)

        # Save obss preprocessor vocabulary and model
        x = utils.get_model_dir(args.model)
        x = utils.get_log_dir(args.model)
        if args.save_interval > 0 and status['i'] % args.save_interval == 0:
            with open(status_path, 'w') as dst:
                json.dump(status, dst)
                utils.save_model(acmodel, args.model)


if __name__ == '__main__':
    main()
