import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from datetime import datetime


def read_csv(p):
    df = pd.read_csv(p)
    def epoch_time(row):
        Y, M, D, H, Min, Sec = None, None, None, None, None, None
        try:
            utc_time = datetime.strptime(row['time'], "%Y-%m-%d")
            Y, M, D = utc_time.year, utc_time.month, utc_time.day
        except ValueError:
            utc_time = datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
            Y, M, D, H, Min, Sec = utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute, utc_time.second
        row['year'] = Y
        row['month'] = M
        row['day'] = D
        row['hour'] = H
        row['minute'] = Min
        row['second'] = Sec
        return row # (utc_time - datetime(1970, 1, 1)).total_seconds()
    df = df.apply(epoch_time, axis=1)
    return df

class StockEnv(gym.Env):

    def __init__(self, stock_trend_filepaths=None, obs_column_names=None, w=10, c=5):
        self.stock_info = [read_csv(p) for p in stock_trend_filepaths]
        self.obs_column_names = obs_column_names
        self.obs_size = len(obs_column_names)
        self.horizon = 140
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(float('inf'), float('inf'), 
                                            dtype=float, shape=(self.obs_size*c,)), 

        # self.agents_has_money = True # spend all of the money each time
        self.starting_candle_ind = None
        self.current_candle_ind = None
        self.current_stock_ind = None
        self.num_candles = None
        self.w = w
        self.c = c
        self._seed()
        self.L = self.c * len(self.obs_column_names)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        actions: 
        - 0: buy or keep
        - 1: sell
        """
        if action == 0:
            reward_sign = 1
        else:
            reward_sign = -1
        start_candle = self.stock_info[self.current_stock_ind][self.obs_column_names].iloc[self.current_candle_ind]
        end_candle = self.stock_info[self.current_stock_ind][self.obs_column_names].iloc[self.current_candle_ind + self.w]
        reward = (end_candle['close'] - start_candle['close']) / start_candle['close'] * reward_sign
        self.current_candle_ind += 1
        done = (self.current_candle_ind - self.starting_candle_ind) >= self.horizon or self.current_candle_ind >= (self.num_candles - self.w)# reach the horizon
        info = {}
        return self._get_obs(), reward, done, info

    def reset(self):
        # Choose the excel randomly
        self.current_stock_ind = np.random.randint(0, len(self.stock_info))
        self.num_candles = len(self.stock_info[self.current_stock_ind])
        self.starting_candle_ind = np.random.randint(0, self.num_candles - self.w)
        self.current_candle_ind = self.starting_candle_ind
        return self._get_obs()

    def _get_obs(self):
        i = np.max(self.current_candle_ind - self.c, 0)
        obs = self.stock_info[self.current_stock_ind][self.obs_column_names].iloc[i:self.current_candle_ind].to_numpy().reshape(-1)
        if len(obs) < self.L:
            obs = np.concatenate((np.zeros(self.L - len(obs),), obs))
        return obs
        

