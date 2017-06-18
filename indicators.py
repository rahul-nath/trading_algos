import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

def normalize(df):
    """Normalizes a pandas dataframe"""
    df.fillna(value=1, inplace=True) 
    return df/df.ix[0, :] #(df - df.mean())/df.std() #

def price_sma(price, lookback):
    # need to also plot sma and price
    # normalize the data
    # for all of these, subtract the mean from each 
    # value and divide by the standard deviation
    sma = pd.rolling_mean(price, window=lookback, 
                          min_periods=lookback)
    ratio = price / sma
    return sma, ratio

def bollinger_bands(price, sma, lookback):
    # probably need to also plot sma and price

    rolling_std = pd.rolling_std(price, window=lookback, min_periods=lookback)
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (price - bottom_band) / (top_band - bottom_band)
    return top_band, bottom_band, bbp

def williams_pr(price, lookback):
    lowest_lows = pd.rolling_min(price, window=lookback, min_periods=lookback)
    highest_highs = pd.rolling_max(price, window=lookback, min_periods=lookback)
                                   
    williams_pr = price.copy()
    williams_pr = -100*(highest_highs - williams_pr) / \
                  (highest_highs - lowest_lows)

    return williams_pr

def get_indicators(prices, sd, ed, symbols, start_val, lookback):
    # There is no limit on leverage.
    # Allowable positions are: 200 shares long, 
    # 200 shares short, 0 shares

    dates = pd.date_range(sd, ed)

    sma, psr = price_sma(prices.copy(), lookback)
    top, bottom, bbp = bollinger_bands(prices.copy(), sma.copy(), lookback)
    wpr = williams_pr(prices, lookback)

    # normalize wpr?

    sma = normalize(sma)
    psr = normalize(psr)
    top, bottom, bbp = normalize(top), normalize(bottom), normalize(bbp)

    psr.rename(columns={'AAPL':'Price/SMA Ratio'}, inplace=True)
    bbp.rename(columns={'AAPL':'BBP'}, inplace=True)
    top.rename(columns={'AAPL':'Top BB'}, inplace=True)
    bottom.rename(columns={'AAPL':'Bottom BB'}, inplace=True)
    sma.rename(columns={'AAPL':'SMA'}, inplace=True)
    wpr.rename(columns={'AAPL':'Williams %R'}, inplace=True)

    return psr, bbp, top, bottom, sma, wpr
    
def get_stats(port_val, s):
    # Get daily portfolio value
    daily_returns = (port_val/port_val.shift(1)) - 1
    # Get portfolio statistics
    cr = (port_val[-1]/port_val[0]) - 1
    print "cum return of ", s, cr
    adr = daily_returns.mean()
    print "mean of daily return of ", s, adr
    sddr = daily_returns.std()
    print "std of daily return of ", s, sddr
    return daily_returns, cr, adr, sddr
