"""
Obtained from https://github.com/vermouth1992/drl-portfolio-management/blob/master/src/environment/portfolio.py
"""

from __future__ import print_function

import numpy as np
import pandas as pd

import gym
import gym.spaces
from data.utils import read_data, sharpe, max_drawdown, plot_portfolio_value, plot_cost, plot_return

import os
import sys
fileDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(fileDir)

class DataProvider(object):
    """
    this class is to provide data for new episode
    """

    def __init__(self, dataset_path, window_size=50, steps=730):
        """
        :param dataset_path: the path of dataset
        :param steps: the total number of steps in each episode, default is 2 years
        :param window_size:  observation window, the length of trading period before current
        """
        #data: (num_stock, len(all time), [open, high, low, close])
        #time: a list of string type time;
        #asset_name: a list of length num_stocks with assets name;
        self.data, self.time, self.asset_name = read_data(dataset_path)
        self.window_size = window_size
        self.steps = steps

        # change the start date
        start_idx = 3000
        self.data = self.data[:, int(start_idx):, :]
        self.time = self.time[int(start_idx): ]

        self.reset()

    def f_step(self):
        # get the observation data window for the next step
        self.step += 1

        data_window = self.data[:, self.step:self.step + self.window_size, :].copy()
        time_window = self.time[self.step:self.step + self.window_size].copy()

        # add cash position: obs: (num_stock + 1, window_size, 4)
        cash_position = np.ones((1, self.window_size, data_window.shape[2]))
        obs = np.concatenate((cash_position, data_window), axis=0)

        done = bool(self.step > self.steps)
        return obs, time_window, done

    def reset(self):
        self.step = 0

        # normalize the data by closing price !!! wait to be added
        init_data = self.data[:, self.step:self.step + self.window_size, :].copy()
        init_time = self.time[self.step:self.step + self.window_size].copy()
        cash_position = np.ones((1, self.window_size, init_data.shape[2]))
        obs = np.concatenate((cash_position, init_data), axis=0)
        return obs, init_time


class PortfolioInfo(object):
    """
    this class is to get the performance of the portfolio under the new action
    """

    def __init__(self, asset_names, steps, trading_cost=0.001):
        """
        :param asset_names: the list of asset names
        :param steps: the total number of steps in each episode, default is 2 years
        :param trading_cost: the cost for selling or purchasing the assets
        """

        self.asset_names = asset_names
        self.steps = steps
        self.cost = trading_cost
        self.reset()

    def f_step(self, w1, y1):
        """
        :param w1: new action of portfolio weights, dim = asset_dim
        :param y1: price relative vector close/open
        """

        eps = 1e-8     # to avoid be divided by 0
        p0 = self.p0
        w0 = self.w0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into
        mu1 = self.cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio
        assert mu1 < 1, 'Cost is larger than current holding'

        p1 = (1-mu1) * p0 * np.dot(y1, w0)  # (eq8)(eq2) portfolio value
        p1 = np.clip(p1, 0, np.inf)  # no shorts

        rho1 = p1 / p0 - 1  # rate of returns (eq9)
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return (eq10)
        immediate_reward = r1 / self.steps * 1000  # immediate reward, multiplied by 1000 be more convenient

        # remember for next step
        self.w0 = w1
        self.p0 = p1
        # if we run out of money, we're done (losing all the money)
        done = bool(p1 == 0)

        info = {"immediate_reward": immediate_reward,
                "rate_of_return": rho1,
                "log_return": r1,
                "portfolio_value": p1,
                "cost": mu1,
                "pre_weight": w0,
                "weight": w1,
                "market_return": y1.mean(),
                "best_return": max(y1)}

                # "weights_mean": w1.mean().astype(float),
                # "weights_std": w1.std().astype(float)

        #for i, name in enumerate(['cash'] + self.asset_names):
        #    info['weight_' + name] = w1[i].astype(float)
        #    info['price_' + name] = y1[i].astype(float)

        self.infos.append(info)
        return immediate_reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0    # the initial portfolio value
        self.w0 = np.array([1.0] + [0.0] * (len(self.asset_names)))   # the initial action


class PortfolioEnv(gym.Env):

    def __init__(self, dataset_path, window_size=50, steps=730, trading_cost=0.0025):
        """
        :param dataset_path: the path of dataset
        :param steps: the total number of steps in each episode, default is 2 years
        :param window_size: observation window, the length of trading period before current
        :param trading_cost: the cost for selling or purchasing the assets
        """

        self.window_size = window_size
        self.provider = DataProvider(dataset_path, self.window_size, steps)
        self.asset_dim = len(self.provider.asset_name) + 1
        self.feature_dim = self.provider.data.shape[-1]

        self.portfolio_info = PortfolioInfo(self.provider.asset_name, steps, trading_cost)

        # action space: the portfolio weights from 0 to 1 for each asset, including cash
        self.action_space = gym.spaces.Box(0, 1, shape=(self.asset_dim,), dtype=np.float32)
        # get the observation space, shape = [asset_dim, window_size, [open, high, low, close]]
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.asset_dim, self.window_size,self.feature_dim),
                                          dtype=np.float32),
            'weights': self.action_space})

        self.reset()

    def f_step(self, obs1, action):
        """
        this funciton is to get the portfolo infomation given the new action
        :param obs1: the observation with the lastest price relative vector y1
        :param action: the new action, a1
        :return: immediate reward, the dict of information and done
        """
        eps = 1e-8
        # normalise just in case, in [0, 1]
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + eps)

        # relative price vector of last observation day (close/open) # including cash position
        close_price = obs1[:, -1, 3]
        open_price = obs1[:, -1, 0]
        y1 = close_price / open_price

        # here done2 is because that the cost is larger than the portfolio value
        reward, info, done2 = self.portfolio_info.f_step(action, y1)
        # calculate return for buy and hold uniformly of each asset
        info['market_value'] = np.cumprod([inf["market_return"] for inf in self.infos + [info]])[-1]
        info['best_value'] = np.cumprod([inf["best_return"] for inf in self.infos + [info]])[-1]
        info['steps'] = self.provider.step  # no. steps
        info["obs1"] = obs1

        # the next state
        obs2, time_window, done1 = self.provider.f_step()
        info["obs2"] = obs2
        # the time for period 1, corresponding to y1
        info['time'] = time_window[-2]

        self.infos.append(info)

        return reward, info, done1 or done2

    def reset(self):
        """
        reset the environment
        :return: the dict of information
        """
        self.infos = []
        self.portfolio_info.reset()
        obs, init_time = self.provider.reset()

        info = {}
        info["observation"] = obs
        info["time"] = init_time[-1]
        info["weight"] = self.portfolio_info.w0

        return info

    def plot(self, portfolio_value = True, cost = True, rate_of_return = True):
        df_info = pd.DataFrame(self.infos)
        df_info['time'] = pd.to_datetime(df_info['time'].values, format='%Y-%m-%d')
        #df_info.set_index('time', inplace=True)

        if portfolio_value:
            plot_portfolio_value(df_info)
        if cost:
            plot_cost(df_info)
        if rate_of_return:
            plot_return(df_info)

    def save_info(self, info_path):
        df_info = pd.DataFrame(self.infos)
        df_info['time'] = pd.to_datetime(df_info['time'].values, format='%Y-%m-%d')

        df = df_info[["time", "rate_of_return", "log_return", "portfolio_value","cost"
                       "market_return", "best_return", "market_value", "best_value"]]

        df.to_csv(info_path)


    def calculation(self):
        df_info = pd.DataFrame(self.infos)

        max_dd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return, self.window_size)

        return df_info, max_dd, sharpe_ratio


if __name__ == "__main__":

    dataset_path = fileDir + '/data/data1130'
    window_size = 5
    steps = 10
    asset_names = ["V"]
    trading_cost = 0.0025


    provider = DataProvider(dataset_path, window_size, steps)

    portfolio_info = PortfolioInfo(asset_names, steps, trading_cost)

    env = PortfolioEnv( dataset_path,
                        window_size=window_size,
                        steps=steps,
                        trading_cost=trading_cost)

    info1 = env.reset()

    # for i in range(3):
       # action = env.action_space.sample()
       # action /= action.sum()
        #reward, info2, done = env.f_step(action)
        #assert not done, "shouldn't be done after %s steps" % i

    df_info = pd.DataFrame(env.infos)
    #final_value = df_info.portfolio_value.iloc[-1]
    #market_value = df_info.market_value.iloc[-1]

