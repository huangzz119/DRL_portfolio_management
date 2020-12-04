import pandas as pd
import json
from data.utils import sharpe, max_drawdown
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np

import os
import sys
fileDir = os.getcwd()

df_cnn = pd.read_csv(fileDir +"/info_cnn.csv")
df_rnn = pd.read_csv(fileDir +"/info_rnn.csv")

df_train_cnn = pd.read_csv(fileDir +"/train_info_cnn.csv")
df_train_rnn = pd.read_csv(fileDir +"/train_info_rnn.csv")

df_rnn["time"] = pd.to_datetime(df_rnn["time"])
df_cnn["time"] = pd.to_datetime(df_cnn["time"])

df_rnn["log_market_return"] = np.log( df_rnn["market_return"] )
df_rnn["log_best_return"] = np.log( df_rnn["best_return"] )


with plt.style.context(['science']):
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.plot(range(len(df_train_rnn))[:30], df_train_rnn.q_value[:30],color="#FF69B4", label="ddpg-rnn")
    plt.plot(range(len(df_train_rnn))[:30], df_train_cnn.q_value[:30], label="ddpg-cnn")

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Q-value', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(linestyle=':')
    plt.xticks(rotation=70, size=9)
    plt.yticks(size=9)
    plt.savefig("train0.eps")
    plt.show()

with plt.style.context(['science']):
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.plot(range(len(df_train_rnn))[:30], df_train_rnn.loss[:30], color="#FF69B4", label="ddpg-rnn")
    plt.plot(range(len(df_train_rnn))[:30], df_train_cnn.loss[:30], label="ddpg-cnn")

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(linestyle=':')
    plt.xticks(rotation=70, size=9)
    plt.yticks(size=9)
    plt.savefig("loss0.eps")
    plt.show()


with plt.style.context(['science']):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 設置x軸主刻度顯示格式（日期）
    #plt.gca().xaxis.set_major_locator(mdate.MonthLocator(interval=1))

    plt.plot(df_rnn['time'][-100:], (df_rnn["rate_of_return"].values+1)[-100:], label="ddpg-rnn")
    plt.plot(df_cnn['time'][-100:], (df_cnn["rate_of_return"].values+1)[-100:], color="#FF69B4", label="ddpg-cnn")
    plt.plot(df_cnn['time'][-100:], (df_cnn["market_return"].values)[-100:], "y", label="uniform")
    plt.plot(df_cnn['time'][-100:], (df_cnn["best_return"].values)[-100:],color="#696969", label="best")


    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(linestyle=':')
    plt.xticks(rotation=70, size=9)
    plt.yticks(size=9)
    plt.savefig("return0.eps")
    plt.show()

with plt.style.context(['science']):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 設置x軸主刻度顯示格式（日期）
    #plt.gca().xaxis.set_major_locator(mdate.MonthLocator(interval=1))

    plt.plot(df_rnn['time'][-50:], (np.cumsum(df_rnn["log_return"]))[-50:], label="ddpg-rnn")
    plt.plot(df_cnn['time'][-50:], (np.cumsum(df_cnn["log_return"]))[-50:], color="#FF69B4", label="ddpg-cnn")
    plt.plot(df_cnn['time'][-50:], (np.cumsum(df_rnn["log_market_return"]))[-50:], "y", label="uniform")
   # plt.plot(df_cnn['time'][-50:], (np.cumsum(df_rnn["log_best_return"]))[-50:],color="#696969", label="best")

    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Accumulate log return', fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.grid(linestyle=':')
    plt.xticks(rotation=70, size=13)
    plt.yticks(size=13)

    plt.savefig("log_return0.eps")
    plt.show()


with plt.style.context(['science']):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 設置x軸主刻度顯示格式（日期）
    #plt.gca().xaxis.set_major_locator(mdate.MonthLocator(interval=1))

    plt.plot(df_rnn['time'][-1000:], (df_rnn["portfolio_value"])[-1000:], label="ddpg-rnn")
    plt.plot(df_cnn['time'][-1000:], (df_cnn["portfolio_value"])[-1000:], color="#FF69B4", label="ddpg-cnn")
    plt.plot(df_cnn['time'][-1000:], (df_rnn["market_value"])[-1000:], "y", label="uniform")
   # plt.plot(df_cnn['time'][-50:], (np.cumsum(df_rnn["log_best_return"]))[-50:],color="#696969", label="best")

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio value', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(linestyle=':')
    plt.xticks(rotation=70, size=9)
    plt.yticks(size=9)

    plt.savefig("portfolio_value0.eps")
    plt.show()
