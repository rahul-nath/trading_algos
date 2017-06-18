"""
(c) 2017 Rahul Nath

This program learns a trading policy using a QLearner
(Reinforcement Learning algorithm). It uses financial
indicators that are discretized into bins.

"""

import datetime as dt
import QLearner as ql
import pandas as pd
import numpy as np
import util as ut
import indicators as ind

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.learner = None
        self.verbose = verbose
        self.indicators = None

    
    def discretize_index(self, df, steps=10, sym="AAPL"):
        df.replace(to_replace=0, value=np.nan, inplace=True)
        df.replace(to_replace=-100, value=np.nan, inplace=True)
        cuts = pd.qcut(df[sym], steps-1, labels=[i for i in xrange(1, steps)])
        cuts.replace(to_replace=np.nan, value=0, inplace=True)
        return cuts
        

    def simulate_trade(self, action, holdings):
        shares = 0
        # hold
        if action == 0:
            if holdings[-1] == -200:
                holdings.append(holdings[-1] + 200)
                shares = 200
            elif holdings[-1] ==  200:
                holdings.append(holdings[-1] - 200)
                shares = -200                
            else:
                holdings.append(0)
        # long
        elif action == 1:
            if holdings[-1] == -200:
                holdings.append(holdings[-1] + 400)
                shares = 400
            elif holdings[-1] ==  0:
                holdings.append(holdings[-1] + 200)
                shares = 200                
            else:
                holdings.append(200)
        # short
        elif action == 2:
            if holdings[-1] == 200:
                holdings.append(holdings[-1] - 400)
                shares = -400
            elif holdings[-1] == 0:
                holdings.append(holdings[-1] - 200)
                shares = -200
            else:
                holdings.append(-200)
        return shares, holdings

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "AAPL", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 100000): 

        # need to determine a lookback and pull data for before hand
        self.learner = ql.QLearner(num_actions=3, num_states=3000)

        # example usage of the old backward compatible util function
        syms=[symbol]
        lookback = 14
        indicator_start = sd.date() - dt.timedelta(days=(lookback*2))
        dates = pd.date_range(indicator_start, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        sma, psr = ind.price_sma(prices, lookback)
        _, _, bbp = ind.bollinger_bands(prices, sma, lookback)
        wpr = ind.williams_pr(prices, lookback)
        psr = psr.ix[sd:, :].dropna()
        wpr = wpr.ix[sd:, :].dropna()
        bbp = bbp.ix[sd:, :].dropna()
        prices = prices.ix[sd:, :]
        self.indicators = pd.concat([self.discretize_index(wpr, sym=symbol), self.discretize_index(bbp, sym=symbol), self.discretize_index(psr, sym=symbol)],axis=1)

        iterations = 5
        convergence_count = 0
        last_return = 0
        daily_returns = (prices.shift(-1)/prices) - 1

        while iterations:
            actions = []
            holdings = [0]
            port_val = [sv]
            state = int(str(1) + "".join([str(int(i)) for i in self.indicators.iloc[0]]))
            actions.append(self.learner.querysetstate(state))
            #yday_return = 
            for day in prices.index:
                # reward based on yesterday's action
                # change in price bewteen first and second day
                if len(holdings) == 1:
                    reward = 0
                elif actions[-1] == 0:
                    if daily_returns.ix[day, 0] > 0 and holdings[-1] > 0:
                        reward = 1000*daily_returns.ix[day, 0]
                    elif daily_returns.ix[day, 0] > 0 and holdings[-1] < 0:
                        reward = -1000*daily_returns.ix[day, 0]
                    elif daily_returns.ix[day, 0] < 0 and holdings[-1] < 0:
                        reward = -1000*daily_returns.ix[day, 0]
                    elif daily_returns.ix[day, 0] < 0 and holdings[-1] > 0:
                        reward = 1000*daily_returns.ix[day, 0]
                elif actions[-1] == 1:
                    # if we correctly picked a buy
                    reward = 1000*daily_returns.ix[day, 0]
                elif actions[-1] == 2:
                    reward = -1000*daily_returns.ix[day, 0]

                #we're calculating the new state for the second day
                hold_ind = 1
                if holdings[-1] != 0:
                    hold_ind = 2
                state = int(str(hold_ind) + "".join([str(int(i)) for i in self.indicators.ix[day, :]]))

                # this will be "yesterday's action" and the price change for second and third day
                actions.append(self.learner.query(state, reward))
                shares, holdings = self.simulate_trade(actions[-1], holdings)

            iterations -= 1
            if self.verbose: print port_val[-1]

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "AAPL", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 100000):

        lookback = 14
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        df_trades = trades.copy()#[1:]
        # get all the indices on trades
        # drop the indices in indicators that are not within trades


        holdings = [0]
        port_val = [sv]
        actions = []
        # first off you're not holding
        state = int(str(1) + "".join([str(int(i)) for i in self.indicators.iloc[0]]))
        actions.append(self.learner.querysetstate(state))
        j = 0
        for day in trades.index:

            # do trade
            #today_price = trades.ix[day, 0]
            shares, holdings = self.simulate_trade(actions[-1], holdings)
            df_trades.ix[day, 0] = shares
            #port_val.append(port_val[-1] + (-shares)*today_price)

            # get today's state
            hold_ind = 1
            if holdings[-1] != 0:
                hold_ind = 2
            state = int(str(hold_ind) + "".join([str(int(i)) for i in self.indicators.iloc[j]]))
            actions.append(self.learner.querysetstate(state))
            # get new action

            j += 1
            #if j >= len(trades.index): break
        if self.verbose: print port_val[-1]    
        # you might want to set the rar for the run here
        # return the resulting trades
        #print df_trades

        return df_trades


if __name__=="__main__":
    q = StrategyLearner(verbose=False)
    q.addEvidence()
    q.testPolicy()
