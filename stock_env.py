import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from datetime import datetime


def read_csv(p):
    df = pd.read_csv(p)
    def epoch_time(t):
        try:
            utc_time = datetime.strptime(t, "%Y-%m-%d")
        except ValueError:
            utc_time = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        return (utc_time - datetime(1970, 1, 1)).total_seconds()
    df['time'] = df['time'].apply(epoch_time)
    return df

class StockEnv(gym.Env):

    def __init__(self, stock_trend_filepaths=None, obs_column_names=None, w=10):
        self.stock_info = [read_csv(p) for p in stock_trend_filepaths]
        self.obs_column_names = obs_column_names
        self.obs_size = len(obs_column_names)
        self.horizon = 140
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(float('inf'), float('inf'), 
                                            dtype=float, shape=(self.obs_size,)), 

        # self.agents_has_money = True # spend all of the money each time
        self.starting_candle_ind = None
        self.current_candle_ind = None
        self.current_stock_ind = None
        self.num_candles = None
        self.w = w
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        actions: 
        - 0: buy or keep
        - 1: sell
        """
        # if action == 0:
        #     self.agents_has_money = False
        # else:
        #     self.agents_has_money = True
        start_candle = self.stock_info[self.current_stock_ind][self.obs_column_names].iloc[self.current_candle_ind]
        end_candle = self.stock_info[self.current_stock_ind][self.obs_column_names].iloc[self.current_candle_ind + self.w]
        reward = (end_candle['close'] - start_candle['close']) / start_candle['close'] # -int(self.agents_has_money) * 
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
        return self.stock_info[self.current_stock_ind][self.obs_column_names].iloc[self.current_candle_ind]