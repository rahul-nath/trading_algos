"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    
    lev_limit = 1.5
    df_val = None
    re_leverage = True

    df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, 
                            usecols=['Date', 'Symbol', 'Order', 'Shares'], na_values=['nan'])
    df_orders.sort_index(inplace=True)

    syms = list(df_orders["Symbol"].unique())
    dates = pd.date_range(df_orders.index.min(), df_orders.index.max())

    while re_leverage:
        df_prices = get_data(syms, dates)
        df_prices = df_prices.drop('SPY', 1)
        df_prices['Cash'] = 1.0
        dates = df_prices.index

        df_trades = pd.DataFrame(0, index=dates, columns=list(df_prices))
        df_trades['Cash'] = 0

        df_holdings = pd.DataFrame(0, index=dates, columns=list(df_prices))
        df_holdings['Cash'] = start_val

        for date, row in df_orders.iterrows():
            shares = 0
            if row['Order'] == "SELL":
                shares = -row['Shares']
            else:
                shares = row['Shares']
            df_trades.ix[date, 'Cash'] += (-shares) * df_prices.ix[date, row['Symbol']]
            df_trades.ix[date, row['Symbol']] += shares

        df_holdings.iloc[0] += df_trades.iloc[0]
        for row in range(1, df_trades.shape[0]):
            df_holdings.iloc[row] = df_trades.iloc[row] + df_holdings.iloc[row-1]

        df_val = df_prices*df_holdings
        re_leverage = False

        for row in range(1, df_val.shape[0]):
            lev = df_val.ix[row, :-1].abs().sum()/(df_val.ix[row, :-1].sum() + df_val.ix[row, 'Cash'])
            if lev > lev_limit:
                df_orders = df_orders.drop(df_trades.index[row])
                re_leverage = True
                break


    #print df_val
    return df_val.sum(axis=1)

def author():
    return 'rnath9'

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()
