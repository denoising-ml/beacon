import study.module.module_datasets as datasets

import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas_datareader as web

"""
pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
"""
from mpl_finance import candlestick_ohlc

def simple_plot(df):
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.plot(df.index, df['close'])
    plt.show()


def candle_plot(df):
    ax = plt.subplot()
    bars = zip(mdates.date2num(df.index), df['open'], df['high'], df['low'], df['close'])
    candlestick_ohlc(ax, bars, width=0.6)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()

if __name__ == "__main__":
    df = datasets.load_HSI()
    print(df.describe())

    candle_plot(df)
