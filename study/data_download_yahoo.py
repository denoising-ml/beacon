'''
Author - Nitin Rana
Date - June 19, 20119
Purpose - Downloads Historical Data from Yahoo Finance on Daily Frequency
Pre-Requisite - You must install pandas-datareader from Python Repository as it not part of standard pandas installation
pip install pandas_datareader #1 Install through Pip
if it fails then use conda
conda install pandas_datareader #Install through Conda

arguments:
ticker The ticker symbol to download historical data for
start Start day in form [Year,Month,Day]

optional arguments:
end End day in form [Year,Month,Day]
interval Interval to fetch historical data (can be 1d, 1wk, 1mo, defaults to 1d)

Usage Examples:

Following Tickers are for References for our Data Download
^N225 = Nikkei 225,
^FTSE = UK FTSE
SI=F = Silver Spot in USD
GC=F = GOLD in USD
CL=F = West Texas Intermediate Crude OIL in USD
^RUT = Russel 2000 Index
^IXIC = NASDAQ Composite Index (US Equity)
^DJI = Dow Jones Industrial Index for US Equity
^GSPC = S&P 500
^HSI = Hang Seng Index

    tickerDict = {
        "^HSI": "HangSeng",
        "^N225": "Nikkei225",
        "^IXIC": "Nasdaq",
        "^DJI": "DJIA",
        "^GSPC": "SP500"
    }

Columns to Create in the Output File
close, open, high, low, Volume --Raw Columns
EMA20, MA10, MA5, --Moving Averages
MACD, CCI, ATR, BOLL_MID, --Momentum
MTM12, MTM6
ROC, SMI, WVAD
US Dollar Index
HIBOR

    tickerDict = {
        "^HSI": "HangSeng",
        "DX-Y.NYB": "USDollar_Index",
        "^N225": "Nikkei225",
        "^IXIC": "Nasdaq",
        "^DJI": "DJIA",
        "^GSPC": "SP500"
    }

'''

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as m
from datetime import datetime
import ta as ta
from ta import *

'''

#Moving Average
def SMA(df, n):
    SMA = pd.Series(pd.rolling_mean(df['Close'], n), name = 'MA_' + str(n))
    df = df.join(SMA)
    return df

#Exponential Moving Average
def EMA(df, n):
    EMA = pd.Series(pd.ewma(df['Close'], span=n, min_periods=n-1), name='EMA_' + str(n))
    df = df.join(EMA)
    return df
'''

def get_market_data_fromfile():
    #This function will download entire S&P 500 Data.
    #Must be Yahoo Tickers, Frequecy is Daily
    #You must have csv file to include all ticker names
    # Column Names are Symbol, Name, Sector
    #S&P 500 Full Data List

    tickers_df = pd.read_csv("sp500_constituents.csv")
    for row in tickers_df.itertuples():
        print(row.Symbol, row.Name, row.Sector)
        start = dt.datetime(2001, 1, 1)
        end = dt.datetime.today()

        if not os.path.exists('marketdata'):
            os.makedirs('marketdata')

        try:
            df = web.get_data_yahoo(row.Symbol, start, end)
            filenameSuffix = row.Symbol  + "_" + row.Sector + "___"+ \
                             "_" + str(start.year) + ("00" + str(start.month))[-2:] + ("00" + str(start.day))[-2:] \
                             + "_" + str(end.year) + ("00" + str(end.month))[-2:] + ("00" + str(end.day))[-2:]

            filenameSuffix_Orig = filenameSuffix + "_Original"

            df.to_csv('marketdata/{}.csv'.format(filenameSuffix_Orig))

            print(row.Symbol, "  ....downloaded")
        except Exception as e:
            print(e, "error")


def get_market_data():
    #Setting up Ticker Dictionary. Key is the Ticker and Value is the file Name
    #Must be Yahoo Tickers, Frequecy is Daily
    tickerDict = {
        "^N225": "Nikkei225",
    }

    #S&P 500 Full Data List
    tickers = pd.read_csv("sp500_constituents.csv")

    for row in tickers.head().itertuples():
        print(row.Index, row.date, row.delay)

    start = dt.datetime(2019, 1, 1)
    end = dt.datetime.today()

    if not os.path.exists('marketdata'):
        os.makedirs('marketdata')

    for ticker in tickerDict:
        print(ticker)

        try:
            df = web.get_data_yahoo(ticker, start, end)
            filenameSuffix = tickerDict.get(ticker)  + \
                             "_" + str(start.year) + ("00" + str(start.month))[-2:] + ("00" + str(start.day))[-2:] \
                             + "_" + str(end.year) + ("00" + str(end.month))[-2:] + ("00" + str(end.day))[-2:]

            filenameSuffix_Orig = filenameSuffix + "_Original"
            df['MA5'] = pd.Series(df['Close']).rolling(5).mean()
            df['MA10'] = pd.Series(df['Close']).rolling(10).mean()
            df['MACD'] = ta.macd(pd.Series(df['Close']), 12, 26, fillna=False)
            df['CCI'] = ta.cci(df["High"], df["Low"], df["Close"], 20, 0.015, fillna=False)
            df['ATR'] = ta.average_true_range(df["High"], df["Low"], df["Close"], 14, fillna=False)
            df['EMA20'] = ta.ema(df["Close"], 20, fillna=False)

            df.to_csv('marketdata/{}.csv'.format(filenameSuffix_Orig))

            print(ticker, "  ....downloaded")
        except Exception as e:
            print(e, "error")

if __name__ == "__main__":
    get_market_data_fromfile()
    # get_market_data()



