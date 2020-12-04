import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

fileDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(fileDir)
dataset_path = fileDir+'/data/data1130'

def read_data(dataset_path):
    """
    read the stock data and get the informations
    here we don't need to be concerned about the missing data
    :param dataset_path: the path of datas
    :return: shape = (num_asset, date, 4), date and asset_name
    """
    files = os.listdir(dataset_path)
    asset_name = [os.path.splitext(arr)[0] for arr in files]
    data_set = []
    for i in range(len(files)):
        pd1 = pd.read_csv(dataset_path + "/" + files[i])
        price = pd1[["Open", "High", "Low", "Close"]].values
        data_set.append(price)
    time = pd1.Date.values.tolist()
    return np.array(data_set), time, asset_name


def sharpe(returns, window_size = 50, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    eps = 1e-8
    return (np.sqrt(window_size) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    eps = 1e-8
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)

def normalize_state(state):
    close_vactor = state[:, :, -1, -1][0]
    base_close = np.ones(state.shape)
    for i in range(state.shape[1]):
        base_close[:, i] = close_vactor[i]
    norm_state = state/base_close
    return norm_state


def plot_portfolio_value(df):
    with plt.style.context(['science']):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # 設置x軸主刻度顯示格式（日期）
        plt.gca().xaxis.set_major_locator(mdate.MonthLocator(interval=12))

        plt.plot(df['time'], df["portfolio_value"], color="#FF69B4", label="model allocation")
        plt.plot(df['time'], df["market_value"], "y", label="uniform allocation")

        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Portfolio value', fontsize=16)
        plt.legend(fontsize=14, loc='best')
        plt.grid(linestyle=':')
        plt.xticks(rotation=70, size=13)
        plt.yticks(size=13)

        plt.savefig("value.eps")

def plot_cost(df):
    with plt.style.context(['science']):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # 設置x軸主刻度顯示格式（日期）
        plt.gca().xaxis.set_major_locator(mdate.MonthLocator(interval=12))

        plt.plot(df['time'], df["cost"], color="#FF69B4", label="cost")

        plt.xlabel('Date', fontsize=16)
        plt.ylabel('cost value', fontsize=16)
        plt.legend(fontsize=14, loc='best')
        plt.grid(linestyle=':')
        plt.xticks(rotation=70, size=13)
        plt.yticks(size=13)

        plt.savefig("cost.eps")


def plot_return(df):
    with plt.style.context(['science']):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # 設置x軸主刻度顯示格式（日期）
        plt.gca().xaxis.set_major_locator(mdate.MonthLocator(interval=12))

        plt.plot(df['time'], df["rate_of_return"]+1, color="#FF69B4", label="model allocation")
        plt.plot(df['time'], df["market_return"], "y", label="uniform allocation")

        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Return', fontsize=16)
        plt.legend(fontsize=14, loc='best')
        plt.grid(linestyle=':')
        plt.xticks(rotation=70, size=13)
        plt.yticks(size=13)

        plt.savefig("return.eps")









