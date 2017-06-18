"""
(c) 2017 Rahul Nath
This program simply generates the daily return
of a portfolio given some allocation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data


def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):
    
    # Read in adjusted closing prices for 
    # given symbols, date range
    dates = pd.date_range(sd, ed)

    # automatically adds SPY
    prices_all = get_data(syms, dates)  

    # only portfolio symbols
    prices = prices_all[syms] 
    prices_SPY = prices_all['SPY']  

    # normalize the first day of values
    normed = prices/prices.ix[0, :]

    # multiply by allocations, 
    # giving relative value of each asset over time
    # and get the real value of the 
    # investment each day over time
    pos_vals = normed*allocs*sv

    # sum across each day (each row)
    # and we the value of the portfolio each day
    port_val = pos_vals.sum(axis=1)

    # Get daily portfolio value
    daily_returns = (port_val/port_val.shift(1)) - 1

    # Get portfolio statistics
    cr = (port_val[-1]/port_val[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    diff = daily_returns - rfr
    sr = np.sqrt(sf)*(diff.mean()/sddr)
    
    # Compare daily portfolio value with 
    # SPY using a normalized plot
    print "Start Date:", sd
    print "End Date:", ed
    print "Symbols:", syms
    print "Allocations:", allocs
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], 
                keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp/df_temp.ix[0, :], 
                  title="Daily portfolio value and SPY")

    # Add code here to properly compute end value
    ev = port_val.ix[-1]
    return cr, adr, sddr, sr, ev

def test_code():

    print "Test 1"
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd=start_date,
        ed = end_date,
        syms = symbols, 
        allocs = allocations,
        sv = start_val, 
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
