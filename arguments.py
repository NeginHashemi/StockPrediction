
import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default='StockPrediction-v0',
                            help="name of the environment to train on (REQUIRED)")
        self.add_argument("--stock_filepaths", type=str, nargs='+')
        self.add_argument("--obs_columns", type=str, nargs='+')
        self.add_argument("--w", type=int, default=10)
        self.add_argument("--c", type=int, default=50)
        self.add_argument("--model", default='model')
        self.add_argument("--seed", type=int, default=1,
                            help="random seed; if 0, a random random seed will be used  (default: 1)")
        self.add_argument("--procs", type=int, default=2,
                            help="number of processes (default: 64)")
        
        # Training arguments
        self.add_argument("--log_interval", type=int, default=10,
                            help="number of updates between two logs (default: 10)")
        self.add_argument('--log_dir', type=str, default=None)
        self.add_argument("--save_interval", type=int, default=1000,
                            help="number of updates between two saves (default: 1000, 0 means no saving)")
        self.add_argument("--frames", type=int, default=int(9e10),
                            help="number of frames of training (default: 9e10)")
        self.add_argument("--patience", type=int, default=100,
                            help="patience for early stopping (default: 100)")
        self.add_argument("--epochs", type=int, default=1000000,
                            help="maximum number of epochs")
        self.add_argument("--frames_per_proc", type=int, default=200,
                            help="number of frames per process before update (default: 40)")
        self.add_argument("--lr", type=float, default=1e-4,
                            help="learning rate (default: 1e-4)")
        self.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
        self.add_argument("--recurrence", type=int, default=400,
                            help="number of timesteps gradient is backpropagated (default: 20)")
        self.add_argument("--optim_eps", type=float, default=1e-5,
                            help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--optim_alpha", type=float, default=0.99,
                            help="RMSprop optimizer apha (default: 0.99)")
        self.add_argument("--batch_size", type=int, default=80,
                                help="batch size for PPO (default: 1280)")
        self.add_argument("--entropy_coef", type=float, default=0.01,
                            help="entropy term coefficient (default: 0.01)")

        # Model parameters
        self.add_argument("--memory_dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--arch", type=str, default='linear',
                            help="Architecture of the policy before AC heads, could be attention")

        # Validation parameters
        self.add_argument("--val_seed", type=int, default=int(1e9),
                            help="seed for environment used for validation (default: 1e9)")
        self.add_argument("--val_interval", type=int, default=1,
                            help="number of epochs between two validation checks (default: 1)")
        self.add_argument("--val_episodes", type=int, default=500,
                            help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")

        # Algo parameters (ours)
        self.add_argument("--discount", type=float, default=0.99,
                            help="discount factor (default: 0.99)")
        self.add_argument("--reward_scale", type=float, default=20.,
                            help="Reward scale multiplier")
        self.add_argument("--gae_lambda", type=float, default=0.99,
                            help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
        self.add_argument("--value_loss_coef", type=float, default=0.5,
                            help="value loss term coefficient (default: 0.5)")
        self.add_argument("--max_grad_norm", type=float, default=0.5,
                            help="maximum norm of gradient (default: 0.5)")
        self.add_argument("--clip_eps", type=float, default=0.2,
                            help="clipping epsilon for PPO (default: 0.2)")
        self.add_argument("--ppo_epochs", type=int, default=4,
                            help="number of epochs for PPO (default: 4)")

        # Model params
        self.add_argument("--state_dim", type=int, default=9)
        self.add_argument("--act_num", type=int, default=3)
        self.add_argument("--flag", type=str, default="stock_ppo")

    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args()

        return args
