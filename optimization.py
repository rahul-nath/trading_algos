"""
MC1-P2: Optimize a portfolio.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the 
# functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for 
    # given symbols, date range
    dates = pd.date_range(sd, ed)

    # automatically adds SPY
    prices_all = get_data(syms, dates)  

    # only portfolio symbols
    prices = prices_all[syms] 
    prices_SPY = prices_all['SPY']  
    normed = prices/prices.ix[0, :] # pass prices        
    start_allocs = [len(syms)*(1.0/len(syms)) for s in syms]

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be 
    # correct for a test case
    
    def min_func(als):
        # normalize the first day of values
        # Get portfolio statistics
        pos_vals = normed*als 
        port_val = pos_vals.sum(axis=1) 
        dr = (port_val/port_val.shift(1)) - 1
        return -(np.sqrt(252.0)*(dr.mean()/dr.std()))

    cons = {'type':'eq', 'fun': lambda x: np.sum(x)-1}
    bnds = [(0,1) for s in syms]
    res = spo.minimize(min_func, start_allocs,
                          method="SLSQP", bounds=bnds, constraints=cons)
    allocs = res['x']
    pos_vals = normed*allocs
    port_val = pos_vals.sum(axis=1) 
    cr = (port_val[-1]/port_val[0]) - 1
    ev = port_val.ix[-1]
    daily_returns = (port_val/port_val.shift(1)) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = np.sqrt(252.0)*(adr/sddr)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], 
                keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp/df_temp.ix[0, :], 
                  title="Daily portfolio value and SPY")
    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are 
    # available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2004,1,1)
    end_date = dt.datetime(2006,1,1)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, 
                                                        ed = end_date,
                                                        syms = symbols, 
                                                        gen_plot = False)
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
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
