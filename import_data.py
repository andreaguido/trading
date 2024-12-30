import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

def datosYahoo(dataframe=None, asset_list=None,
               start="2010-01-01", finish=datetime.today().strftime('%Y-%m-%d'),
               plot=False, log_scale=False):

    dataframe=pd.DataFrame()

    if asset_list is None:
        asset_list = []
    for a in asset_list:
        data = yf.download(a, start=start, end=finish)
        dataframe[a] = data["Close"]

    if plot == True:
        plt.figure(figsize=(12.2, 4.5))
        for i in dataframe.columns.values:
            plt.plot(dataframe[i], label=i)
        plt.title('Price of the Stocks')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price in USD', fontsize=18)
        plt.legend(dataframe.columns.values, loc='upper left')

        if log_scale == True:
            plt.yscale('log')
        plt.show()

    return dataframe
