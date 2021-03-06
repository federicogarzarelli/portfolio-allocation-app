# -*- coding: utf-8 -*-
from datetime import timedelta
import GLOBAL_VARS
import backtrader as bt
import numpy as np
import pandas as pd
import riskparityportfolio as rp
from backtrader.utils.py3 import range
from scipy import stats
import sys
import utils

from risk_budgeting import target_risk_contribution

"""
Custom observer to save the target weights 
"""

class targetweightsobserver(bt.observer.Observer):
    params = (('n_assets', 100),)  # set conservatively to 100 as the dynamic assignment does not work
    lines = tuple(['asset_' + str(i) for i in range(0, params[0][1])])

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    def next(self):
        for asset in range(0, self.params.n_assets):            self.lines[asset][0] = self._owner.weights[asset]

"""
Custom observer to save the effective weights 
"""

class effectiveweightsobserver(bt.observer.Observer):
    params = (('n_assets', 100),)  # set conservatively to 100 as the dynamic assignment does not work
    lines = tuple(['asset_' + str(i) for i in range(0, params[0][1])])

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    def next(self):
        effectiveweights = self._owner.get_effectiveweights()
        for asset in range(0, self.params.n_assets):
            self.lines[asset][0] = effectiveweights[asset]

"""
Custom observer to get dates 
"""


class GetDate(bt.observer.Observer):
    lines = ('year', 'month', 'day',)

    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.year[0] = self._owner.datas[0].datetime.date(0).year
        self.lines.month[0] = self._owner.datas[0].datetime.date(0).month
        self.lines.day[0] = self._owner.datas[0].datetime.date(0).day

"""
Custom indicator to set the minimum period. 
"""


class MinPeriodSetter(bt.Indicator):
    lines = ('dummyline',)

    params = (('period', 180),)

    def __init__(self):
        self.addminperiod(self.params.period)


# Create a subclass of bt.Strategy to define the indicators and logic.
class StandaloneStrat(bt.Strategy):
    # parameters which are configurable for the strategy
    params = (
        ('reb_period', 1),  # every month, we rebalance the portfolio (or every year if freq. is yearly)
        ('lookback_period_short', 60),  # period to calculate the variance
        ('lookback_period_long', 180),  # period to calculate the correlation
        ('moving_average_period', 180),  # period to calcuate the moving average
        ('momentum_period', 180),  # period to calculate the momentum return
        ('momentum_percentile', 0.5),  # percentile of assets with the highest return in a period to form the relative momentum portfolio
        ('initial_cash', 1000000),  # initial amount of cash to be invested
        ('contribution', 0),  # amount of cash to buy invested every month
        ('n_assets', 5),  # number of assets
        ('shares', []), # ticker name of assets used, useful when strats need to have specific assets
        ('shareclass', []),  # class of assets
        ('assetweights', []),  # weights of portfolio items (if provided)
        ('printlog', True),
        ('corrmethod', 'pearson'),  # 'spearman' # method for the calculation of the correlation matrix
        ('leverage',0.0),
        ('data_df', [])
    )

    def __init__(self):
        # To keep track of pending orders and buy price/commission
        self.order = None
        # self.cheating = self.cerebro.p.cheat_on_open
        self.buyprice = None
        self.buycomm = None

        self.weights = [0] * self.params.n_assets
        self.effectiveweights = [0] * self.params.n_assets

        self.startdate = None
        self.timeframe = self.get_timeframe()

        self.leverage = self.params.leverage

        days1month = 1
        days3month = 1
        days6month = 1
        if self.timeframe == "Days":
            self.log("Strategy: you are using data with daily frequency", dt=None)
            days1month = 21
            days3month = 21 * 3
            days6month = 21 * 6
        elif self.timeframe == "Years":
            self.log("Strategy: you are using data with yearly frequency", dt=None)

        self.assets = []  # Save data to backtest into assets, other data (e.g. used in indicators) will not be saved here
        self.assetclose = []  # Keep a reference to the close price
        self.sma = []
        self.momentum = []
        self.mom1month = []
        self.mom3month = []
        self.mom6month = []

        for asset in range(0, self.params.n_assets):
            self.assets.append(self.datas[asset])
            self.assetclose.append(self.datas[asset].close)
            self.sma.append(bt.indicators.MovingAverageSimple(self.datas[asset].close,
                                                              period=self.params.moving_average_period))
            self.momentum.append(bt.indicators.RateOfChange(self.datas[asset].close,
                                                              period=self.params.momentum_period))
            self.mom1month.append(bt.indicators.RateOfChange(self.datas[asset].close,
                                                              period=days1month))
            self.mom3month.append(bt.indicators.RateOfChange(self.datas[asset].close,
                                                              period=days3month))
            self.mom6month.append(bt.indicators.RateOfChange(self.datas[asset].close,
                                                              period=days6month))

        self.benchmark_assets = []  # Save benchmark asset here
        self.benchmark_assetsclose = []  # Keep a reference to the close price
        benchmark_idxs = [i for i, e in enumerate(self.params.shareclass) if e == 'benchmark']
        if benchmark_idxs:
            for benchmark_idx in benchmark_idxs:
                self.benchmark_assets.append(self.datas[benchmark_idx])
                self.benchmark_assetsclose.append(self.datas[benchmark_idx].close)

        self.indassets = []  # Save indicators here
        self.indassetsclose = []  # Keep a reference to the close price
        indicator_idxs = [i for i, e in enumerate(self.params.shareclass) if e == 'non-tradable']
        if indicator_idxs:
            for indicator_idx in indicator_idxs:
                self.indassets.append(self.datas[indicator_idx])
                self.indassetsclose.append(self.datas[indicator_idx].close)

        self.loanassets = [] # Save loan asset here (e.g. margin loan, lombard loan)
        self.loanassetsclose = []  # Keep a reference to the close price
        loan_idxs = [i for i, e in enumerate(self.params.shareclass) if e == 'loan']
        if loan_idxs:
            for loan_idx in loan_idxs:
                self.loanassets.append(self.datas[loan_idx])
                self.loanassetsclose.append(self.datas[loan_idx].close)

        min_period = max(self.params.lookback_period_long, self.params.moving_average_period, self.params.momentum_period)
        MinPeriodSetter(period=min_period)  # Set the minimum period

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() > GLOBAL_VARS.params['startdate']:
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        # dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']
        self.dates_eom_df = dates_eom_df

        # get file with trading days for
        self.trading_days = pd.read_csv('src/trading_days.csv')

    def start(self):
        self.broker.set_fundmode(fundmode=True, fundstartval=100.00)  # Activate the fund mode, default has 100 shares
        self.broker.set_cash(self.params.initial_cash)  # Set initial cash of the account

        if self.timeframe == "Days":
            # Add a timer which will be called on the 20st trading day of the month, when salaries are paid
            self.add_timer(
                bt.timer.SESSION_START,
                monthdays=[1],  # called on the first day of the month
                monthcarry=True,  # called on another day if first day is vacation/weekend)
                timername = 'contribution_timer',
            )
            # self.add_timer(
            #     bt.timer.SESSION_START,
            #     monthdays=[1],  # rebalance the portfolio on the 1st day of the month
            #     monthcarry=True,  # called on another day if 20th day is vacation/weekend)
            #     timername = 'rebalance_timer',
            # )
        elif self.timeframe == "Years":
            # Add a timer which will be called every year
            self.add_timer(bt.timer.SESSION_START,timername = 'contribution_timer',)
            self.add_timer(bt.timer.SESSION_START,timername = 'rebalance_timer',)

    def get_timeframe(self):
        dates = [self.datas[0].datetime.datetime(i) for i in range(1, len(self.datas[0].datetime.array) + 1)]
        datediff = stats.mode(np.diff(dates))[0][0]
        if datediff > timedelta(days=250):
            return "Years"
        elif datediff < timedelta(days=2):
            return "Days"

    def notify_timer(self, timer, when, timername, *args, **kwargs):
        if timername == 'contribution_timer':
            # Add/Remove the cash to the broker
            if self.startdate is not None and self.datas[0].datetime.datetime(0) >= self.startdate: # Start adding removing cash after the minimum period
                if self.params.contribution != 0:
                    if abs(self.params.contribution) < 1: # The user specifies a % amount
                        contribution = self.broker.get_value() * self.params.contribution
                    else: # the user specifies a dollar amount
                        contribution = self.params.contribution
                    if self.timeframe == "Days":
                        contribution = contribution / 12

                    if self.broker.get_value() > 0: # Withdraw money only is value is positive
                        self.broker.add_cash(contribution)  # Add monthly cash on the 20th day OR withdraw money
                        if contribution > 0:
                            self.log('CASH ADDED: portfolio value was %.f. New portfolio value is %.f' % (self.broker.get_value(), (self.broker.get_value() + contribution)))
                        else:
                            self.log('CASH REMOVED: portfolio value was %.f. New portfolio value is %.f' % (self.broker.get_value(), (self.broker.get_value() + contribution)))
        # if timername == 'rebalance_timer':
        #     self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
        #             self.broker.get_cash(),
        #             self.broker.get_value(),
        #             self.broker.get_fundshares(),
        #             self.broker.get_fundvalue()))
        #     if self.strategy_name == 'benchmark':
        #         self.buybenchmark()
        #     else:
        #         self.rebalance()

    def log(self, txt, dt=None):
        ''' Logging function for this strategy txt is the statement and dt can be used to specify a specific datetime'''
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Size: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.size,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    'SELL EXECUTED, Price: %.2f, Size: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.size,
                     order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected.')

        # Write down: no pending order
        self.order = None

    def get_weights(self):
        return self.weights

    def get_effectiveweights(self):
        PortfolioValue = self.broker.get_value()
        for asset in range(0, self.params.n_assets):
            position = self.broker.getposition(data=self.assets[asset]).size
            if len(self.assetclose[asset].get(0)) == 0:
                ThisAssetval = 0
            else:
                ThisAssetval = self.assetclose[asset].get(0)[0] * position
            self.effectiveweights[asset] = ThisAssetval / PortfolioValue
        return self.effectiveweights

    def nextstart(self):
        self.startdate = self.datas[0].datetime.datetime(1) # the first date is the next one, since nextstart is called one bar before next

        #self.initial_buy()

        super(StandaloneStrat, self).nextstart()

    """
    def initial_buy(self):
    # Buy at open price
    for asset in range(0, self.params.n_assets):
        target_size = int(self.broker.get_cash() * self.weights[asset] / self.assetclose[asset].get(0)[0])
        if target_size > 0:
            self.buy(data=self.datas[asset], exectype=bt.Order.Limit,
                     price=self.assetclose[asset].get(0)[0], size=target_size, coo=True)
    """

    def rebalance_flg(self):
        if self.timeframe == "Days":
            rebalance_flg = any(self.dates_eom_df.dt.strftime(date_format="%Y-%m-%d").isin(
                [self.datas[0].datetime.datetime(0).date().strftime(format="%Y-%m-%d")]))
        elif self.timeframe == "Years":
            rebalance_flg = len(self) % self.params.reb_period == 0
        return rebalance_flg

    def rebalance(self):
        sold_value = 0

        for asset in range(0, self.params.n_assets): # Sell long assets at open price
            position = self.broker.getposition(data=self.assets[asset]).size
            if position > 0:
                self.sell(data=self.datas[asset], exectype=bt.Order.Limit,
                          price=self.assetclose[asset].get(0)[0], size=position, coo=True)
                sold_value = sold_value + (self.assetclose[asset].get(0)[0] * position)
        cash_after_sell = sold_value + self.broker.get_cash()

        if self.leverage > 0:  # short sell cash
            position = self.broker.getposition(data=self.loanassets[0]).size
            if position < 0: # first reverse the short cash position
                self.buy(data=self.loanassets[0], exectype=bt.Order.Limit,
                         price=self.loanassetsclose[0].get(0)[0], size=position, coo=True)
                sold_value = (self.loanassetsclose[0].get(0)[0] * position)
                cash_after_sell = cash_after_sell + sold_value
            # then short sell cash according to the leverage ratio
            target_size = int(cash_after_sell * self.leverage / self.loanassetsclose[0].get(0)[0])
            self.sell(data=self.loanassets[0], exectype=bt.Order.Limit,
                      price=self.loanassetsclose[0].get(0)[0], size=target_size, coo=True)
            sold_value = (self.loanassetsclose[0].get(0)[0] * target_size)
            cash_after_sell = cash_after_sell + sold_value

        if cash_after_sell < 0: # If after having sold everything, the portfolio value is negative, close the backtesting
        #    self.env.runstop()
        #    print('Early stop envoked! Portfolio value is %.f.' % cash_after_sell)
            print('Portfolio value is negative %.f.' % cash_after_sell)

        # Buy at open price
        for asset in range(0, self.params.n_assets):
            target_size = int(cash_after_sell * self.weights[asset] / self.assetclose[asset].get(0)[0])
            if target_size > 0:
                self.buy(data=self.datas[asset], exectype=bt.Order.Limit,
                         price=self.assetclose[asset].get(0)[0], size=target_size, coo=True)

    def buybenchmark(self): #TODO: check that benchmark equals to only stocks when benchmark is the stock
        sold_value = 0
        # Sell at open price
        benchmark_idx = self.params.shareclass.index('benchmark')
        position = self.broker.getposition(data=self.benchmark_assets[0]).size
        if position > 0:
            self.sell(data=self.datas[benchmark_idx], exectype=bt.Order.Limit,
                      price=self.benchmark_assetsclose[0].get(0)[0], size=position, coo=True)
            sold_value = sold_value + (self.benchmark_assetsclose[0].get(0)[0] * position)

        cash_after_sell = sold_value + self.broker.get_cash()

        if cash_after_sell < 0: # If after having sold everything, the portfolio value is negative, close the backtesting
        #    self.env.runstop()
        #    print('Early stop envoked! Portfolio value is %.f.' % cash_after_sell)
            print('Portfolio value is negative %.f.' % cash_after_sell)

        # Buy at open price
        target_size = int(cash_after_sell * 1 / self.benchmark_assetsclose[0].get(0)[0])
        if target_size > 0:
            self.buy(data=self.datas[benchmark_idx], exectype=bt.Order.Limit,
                     price=self.benchmark_assetsclose[0].get(0)[0], size=target_size, coo=True)

    def riskparity_nested_weights(self):
        assetclass_allocation = {
            "gold": 1,
            "commodity": 1,
            "equity": 1,
            "bond_lt": 1,
            "bond_it": 1,
            "money_market": 0
        } # if 1 consider for risk parity; if 0 do not

        tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]

        # asset class replacements for this strategy
        # equity_intl --> equity
        # money_market --> bond_it
        # cryptocurrencies --> gold
        # real_estate --> equity
        tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        # tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
        tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
        tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

        rp_shareclass = [x for x in tradable_shareclass if x not in ['money_market']]

        shareclass_prices = []
        shareclass_cnt = []
        assetWeights = []
        # First run risk parity at asset class level
        for key in assetclass_allocation:
            # Get the assets whose asset class is "key"
            count = sum(map(lambda x: x == key, rp_shareclass))
            shareclass_cnt.append(count)
            if count > 1:
                thisAssetClass_target_risk = [1 / count] * count # Same risk for each assetclass = risk parity

                # calculate the logreturns
                thisAssetClass_idx = [i for i, e in enumerate(rp_shareclass) if e == key]
                thisAssetClassClose = [self.assetclose[i].get(size=self.params.lookback_period_long) for i in
                                       thisAssetClass_idx]

                # Check if asset prices is equal to the lookback period for all assets exist
                thisAssetClassClose_len = [len(i) for i in thisAssetClassClose]
                if not all(elem == self.params.lookback_period_long for elem in thisAssetClassClose_len):
                    return

                logrets = [np.diff(np.log(x)) for x in thisAssetClassClose]

                if self.params.corrmethod == 'pearson':
                    corr = np.corrcoef(logrets)
                elif self.params.corrmethod == 'spearman':
                    corr, p, = stats.spearmanr(logrets, axis=1)

                stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in
                                   logrets])  # standard dev indicator
                stddev_matrix = np.diag(stddev)
                cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

                # Calculate the asset class weights
                thisAssetClassWeights = target_risk_contribution(thisAssetClass_target_risk, cov)
                assetWeights.append(thisAssetClassWeights)

                # Calculate the synthetic asset class price
                prod = 0
                for i in range(0, count):
                    prod = np.add(prod, np.multiply(thisAssetClassWeights[i], thisAssetClassClose[i]))

                shareclass_prices.append(prod)

            if count == 1:
                thisAssetClass_idx = [i for i, e in enumerate(rp_shareclass) if e == key]
                thisAssetClassClose = [self.assetclose[i].get(size=self.params.lookback_period_long) for i in
                                       thisAssetClass_idx]
                shareclass_prices.append(thisAssetClassClose[0])
                assetWeights.append(np.asarray([1]))

        # Check if asset prices is equal to the lookback period for all assets exist
        shareclass_prices_len = [len(i) for i in shareclass_prices]
        if not all(elem == self.params.lookback_period_long for elem in shareclass_prices_len):
            return

        # Now re-run risk parity at portfolio level, using the synthetic assets
        target_risk = [1 / np.count_nonzero(shareclass_cnt)] * np.count_nonzero(
            shareclass_cnt)  # Same risk for each assetclass = risk parit
        logrets = [np.diff(np.log(x)) for x in shareclass_prices]

        if self.params.corrmethod == 'pearson':
            corr = np.corrcoef(logrets)
        elif self.params.corrmethod == 'spearman':
            corr, p, = stats.spearmanr(logrets, axis=1)

        stddev = np.array(
            [np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
        stddev_matrix = np.diag(stddev)
        cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

        assetClass_weights = target_risk_contribution(target_risk, cov)

        # Now cascade down the weights at asset level
        keys = pd.DataFrame()
        keys["keys"] = list(assetclass_allocation.keys())
        keys["cnt"] = shareclass_cnt
        keys = keys[keys.cnt > 0]
        keys_lst = keys['keys'].tolist()

        weights = pd.DataFrame(columns=["sort", "weights"])
        for i in range(0, len(assetClass_weights)):
            thisAssetClass_idx = [k for k, e in enumerate(rp_shareclass) if e == keys_lst[i]]
            for j in range(0, len(assetWeights[i])):
                to_append = [thisAssetClass_idx[j], assetClass_weights[i] * assetWeights[i][j]]
                a_series = pd.Series(to_append, index=weights.columns)
                weights = weights.append(a_series, ignore_index=True)

        # and rearrange the weights
        weights_lst = weights.sort_values("sort")["weights"].to_list()

        tradable_shareclass_flg = [assetclass_allocation[k] for k in tradable_shareclass]
        tradable_df = pd.DataFrame(data = np.multiply(np.cumsum(tradable_shareclass_flg), tradable_shareclass_flg), columns=['index'])
        weights_df = pd.DataFrame(data = list(zip(weights_lst,list(np.arange(0, len(weights_lst))+1))), columns=['weights', 'index'])
        weights_df_merged = pd.merge(left = tradable_df, right = weights_df, how = 'left' , left_on=['index'], right_on= ['index'])
        weights_df_merged['weights'] = weights_df_merged['weights'].fillna(0)
        weights_lst = weights_df_merged['weights'].tolist()

        return weights_lst

    def adm_weights(self):
        # Required asset classes to execute the strategy
        assetclasses = ["equity", "equity_intl", "bond_lt"]

        # Get the assets belonging to the categories required and ignore the rest
        strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]

        # Check that only 3 assets are delivered with the shareclasses defined above
        assetclass_cnt = {}
        assetclass_flag = {}
        for assetclass in assetclasses:
            count = sum(map(lambda x: x == assetclass, strat_shareclass))
            assetclass_cnt[assetclass] = count
            if count > 0:
                assetclass_flag[assetclass] = 1
            else:
                assetclass_flag[assetclass] = 0

        for key in assetclass_cnt.keys():
            if not assetclass_cnt[key] == 1:
                sys.exit('Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(key) +
                         '. Accelerating dual moment strategy requires exactly one asset for this category.')

        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            # mom1month.append(self.mom1month[asset][0])
            # mom3month.append(self.mom3month[asset][0])
            # mom6month.append(self.mom6month[asset][0])
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df['score'] = (momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > \
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0]:
            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[
                0]:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 1
            else:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 1
        else:
            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > \
                    adm_indassetsclose[0]:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 1
            else:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 1

        return momentum_df['asset_weight'].to_list()

    def specific_adm_weights(self):
        # Required asset classes to execute the strategy
        shares = ['VFINX', 'VINEX', 'VUSTX']
        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'shares':self.params.shares[0:self.params.n_assets],
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df['score'] = (momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > \
                momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0]:
            if momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > adm_indassetsclose[0]:
                momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 1
            else:
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 1
        else:
            if momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] > adm_indassetsclose[0]:
                momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 1
            else:
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 1

        return momentum_df['asset_weight'].to_list()

    def adm_diversified_weights(self):
        # Required asset classes to execute the strategy
        assetclasses = ["equity", "equity_intl", "bond_it", "bond_lt", "gold", "commodity"]

        # Get the assets belonging to the categories required and ignore the rest
        strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]

        # Check that only 3 assets are delivered with the shareclasses defined above
        assetclass_cnt = {}
        assetclass_flag = {}
        for assetclass in assetclasses:
            count = sum(map(lambda x: x == assetclass, strat_shareclass))
            assetclass_cnt[assetclass] = count
            if count > 0:
                assetclass_flag[assetclass] = 1
            else:
                assetclass_flag[assetclass] = 0

        for key in assetclass_cnt.keys():
            if not assetclass_cnt[key] == 1:
                sys.exit('Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(key) +
                         '. Accelerating dual moment strategy requires exactly one asset for this category.')

        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            # mom1month.append(self.mom1month[asset][0])
            # mom3month.append(self.mom3month[asset][0])
            # mom6month.append(self.mom6month[asset][0])
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df['score'] = (momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > \
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0]:
            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[
                0]:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 1
            else:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7
        else:
            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > \
                    adm_indassetsclose[0]:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 1
            else:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7

        return momentum_df['asset_weight'].to_list()

    def adm_gradient_weights(self):
        # Required asset classes to execute the strategy
        assetclasses = ["equity", "equity_intl", "bond_lt"]

        # Get the assets belonging to the categories required and ignore the rest
        strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]

        # Check that only 3 assets are delivered with the shareclasses defined above
        assetclass_cnt = {}
        assetclass_flag = {}
        for assetclass in assetclasses:
            count = sum(map(lambda x: x == assetclass, strat_shareclass))
            assetclass_cnt[assetclass] = count
            if count > 0:
                assetclass_flag[assetclass] = 1
            else:
                assetclass_flag[assetclass] = 0

        for key in assetclass_cnt.keys():
            if not assetclass_cnt[key] == 1:
                sys.exit('Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(key) +
                         '. Accelerating dual moment strategy requires exactly one asset for this category.')

        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            # mom1month.append(self.mom1month[asset][0])
            # mom3month.append(self.mom3month[asset][0])
            # mom6month.append(self.mom6month[asset][0])
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df['score'] = (momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > \
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0]:

            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] + 0.2:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 1
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] + 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.25
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] - 0.1):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.5
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] - 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] - 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.75
            elif momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] - 0.2:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 1

        else:
            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] + 0.2:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 1
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] + 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.25
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] - 0.1):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.5
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] - 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] - 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.75
            elif momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] - 0.2:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 1

        return momentum_df['asset_weight'].to_list()

    def specific_vigilant_weights(self):
        # Required asset classes to execute the strategy
        assetclasses = ["equity", "equity_intl", "bond_lt", "bond_it"]
        # shares = ['VFINX', 'EFA', 'EEM', 'AGG', 'LQD', 'IEF', 'SHY']
        # defensive_offensive = ["offensive", "offensive", "offensive", "offensive",
        #                        "defensive", "defensive", "defensive"]
        shares = ['SPY', 'EFA', 'FEMKX', 'AGG', 'LQD', 'IEF', 'SHY']
        defensive_offensive = ["offensive", "offensive", "offensive", "offensive", "defensive", "defensive", "defensive"]

        # Get the assets belonging to the categories required and ignore the rest
        strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]
        strat_shares = [x for x in self.params.shares if x in shares]

        defensive_offensive_df = pd.DataFrame(list(zip(shares, defensive_offensive)),
                                              columns=['shares', 'defensive_offensive'])

        # Check that only 3 assets are delivered with the shareclasses defined above
        assetclass_cnt = {}
        assetclass_flag = {}
        for assetclass in assetclasses:
            count = sum(map(lambda x: x == assetclass, strat_shareclass))
            assetclass_cnt[assetclass] = count
            if count > 0:
                assetclass_flag[assetclass] = 1
            else:
                assetclass_flag[assetclass] = 0

        for key in assetclass_cnt.keys():
            if not assetclass_cnt[key] >= 1:
                sys.exit('Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(key) +
                         '. Vigilant asset allocation requires at least one asset for this category.')

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []
        mom12month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            # mom1month.append(self.mom1month[asset][0])
            # mom3month.append(self.mom3month[asset][0])
            # mom6month.append(self.mom6month[asset][0])
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)
            mom12month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-12])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'shares': strat_shares,
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'mom12month': mom12month,
                                    'shares': self.params.shares[0:self.params.n_assets],
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df = pd.merge(left=momentum_df, right=defensive_offensive_df, how='left', left_on=['shares'],
                               right_on=['shares'])

        momentum_df['score'] = 12 * momentum_df['mom1month'] + 4 * momentum_df['mom3month'] + 2 * momentum_df[
            'mom6month'] + momentum_df['mom12month']
        momentum_df['asset_weight'] = 0

        if all(momentum_df.loc[momentum_df['defensive_offensive'] == 'offensive', 'score'] > 0):
            momentum_df_off = momentum_df.where(momentum_df.defensive_offensive == "offensive")
            idx = momentum_df_off.iloc[momentum_df_off['score'].idxmax(axis=0, skipna=True)]['idx']
            momentum_df.loc[momentum_df['idx'] == idx, 'asset_weight'] = 1
        else:
            momentum_df_def = momentum_df.where(momentum_df.defensive_offensive == "defensive")
            idx = momentum_df_def.iloc[momentum_df_def['score'].idxmax(axis=0, skipna=True)]['idx']
            momentum_df.loc[momentum_df['idx'] == idx, 'asset_weight'] = 1

        return momentum_df['asset_weight'].to_list()

    def adm_grad_div_weights(self):
        # Required asset classes to execute the strategy
        assetclasses = ["equity", "equity_intl", "bond_it", "bond_lt", "gold", "commodity"]

        # Get the assets belonging to the categories required and ignore the rest
        strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]

        # Check that only 3 assets are delivered with the shareclasses defined above
        assetclass_cnt = {}
        assetclass_flag = {}
        for assetclass in assetclasses:
            count = sum(map(lambda x: x == assetclass, strat_shareclass))
            assetclass_cnt[assetclass] = count
            if count > 0:
                assetclass_flag[assetclass] = 1
            else:
                assetclass_flag[assetclass] = 0

        for key in assetclass_cnt.keys():
            if not assetclass_cnt[key] == 1:
                sys.exit('Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(key) +
                         '. Accelerating dual moment strategy requires exactly one asset for this category.')

        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            # mom1month.append(self.mom1month[asset][0])
            # mom3month.append(self.mom3month[asset][0])
            # mom6month.append(self.mom6month[asset][0])
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df['score'] = (momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > \
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0]:

            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] + 0.2:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 1
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] + 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 0.75
                # All weather w/o stocks
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7*0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7*0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7*0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7*0.25

            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] - 0.1):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 0.5
                # All weather w/o stocks
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7*0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7*0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7*0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7*0.5

            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] - 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > adm_indassetsclose[0] - 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 0.25
                # All weather w/o stocks
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7*0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7*0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7*0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7*0.75

            elif momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] < adm_indassetsclose[0] - 0.2:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7
        else:
            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] + 0.2:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 1
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] + 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 0.75
                # All weather w/o stocks
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7*0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7*0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7*0.25
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7*0.25

            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] - 0.1):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 0.5
                # All weather w/o stocks
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7*0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7*0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7*0.5
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7*0.5

            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] - 0.1 and
                 momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > adm_indassetsclose[0] - 0.2):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 0.25
                # All weather w/o stocks
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7*0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7*0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7*0.75
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7*0.75

            elif momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] < adm_indassetsclose[0] - 0.2:
                # All weather w/o stocks
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_it', 'asset_weight'] = 0.15*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.075*10/7
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.075*10/7

        return momentum_df['asset_weight'].to_list()

    def specific_adm_grad_div_weights(self):
        # Required asset classes to execute the strategy
        shares = ['VFINX', 'VINEX', 'TLT', 'IEF', 'GLD', 'GSG']
        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'shares':self.params.shares[0:self.params.n_assets],
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df['score'] = (momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > \
                momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0]:

            if momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > adm_indassetsclose[0] + 0.2:
                momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 1
            elif (momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] < adm_indassetsclose[0] + 0.2):
                momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 0.75
                # All weather w/o stocks
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7*0.25
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7*0.25
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7*0.25
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7*0.25

            elif (momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] < adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > adm_indassetsclose[0] - 0.1):
                momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 0.5
                # All weather w/o stocks
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7*0.5
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7*0.5
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7*0.5
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7*0.5

            elif (momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] < adm_indassetsclose[0] - 0.1 and
                 momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > adm_indassetsclose[0] - 0.2):
                momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 0.25
                # All weather w/o stocks
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7*0.75
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7*0.75
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7*0.75
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7*0.75

            elif momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] < adm_indassetsclose[0] - 0.2:
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7
        else:
            if momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] > adm_indassetsclose[0] + 0.2:
                momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 1
            elif (momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] > adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] < adm_indassetsclose[0] + 0.2):
                momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 0.75
                # All weather w/o stocks
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7*0.25
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7*0.25
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7*0.25
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7*0.25

            elif (momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] < adm_indassetsclose[0] + 0.1 and
                 momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] > adm_indassetsclose[0] - 0.1):
                momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 0.5
                # All weather w/o stocks
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7*0.5
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7*0.5
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7*0.5
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7*0.5

            elif (momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] < adm_indassetsclose[0] - 0.1 and
                 momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] > adm_indassetsclose[0] - 0.2):
                momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 0.25
                # All weather w/o stocks
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7*0.75
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7*0.75
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7*0.75
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7*0.75

            elif momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] < adm_indassetsclose[0] - 0.2:
                momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4*10/7
                momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15*10/7
                momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075*10/7
                momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075*10/7

        return momentum_df['asset_weight'].to_list()

    # TODO ADM use new common score and weights functions
    def specific_adm_grad_div_weights_2(self):
        # Required asset classes to execute the strategy
        data = self.params.data_df[['VFINX', 'VINEX', 'TLT', 'IEF', 'GLD', 'GSG','DTB3']]
        reference_date = self.datas[0].datetime.date(0)
        momentum_df = utils.ADM_weights(data, reference_date)

        shares_df = pd.DataFrame(list(zip(self.params.data_df.columns, self.params.shareclass)), columns = ['shares', 'shareclass'])
        shares_df = shares_df.loc[~shares_df.shareclass.isin(['non-tradable', 'benchmark', 'loan'])]
        shares_df_momentum_df = pd.merge(shares_df, momentum_df, how='left', left_on=['shares'], right_on=['shares'])['asset_weight'].fillna(0)

        return shares_df_momentum_df.to_list()

    def gem_weights(self):
        # Required asset classes to execute the strategy
        assetclasses = ["equity", "equity_intl", "bond_lt", "money_market"]

        # Get the assets belonging to the categories required and ignore the rest
        strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]

        # Check that only 4 assets are delivered with the shareclasses defined above
        assetclass_cnt = {}
        assetclass_flag = {}
        for assetclass in assetclasses:
            count = sum(map(lambda x: x == assetclass, strat_shareclass))
            assetclass_cnt[assetclass] = count
            if count > 0:
                assetclass_flag[assetclass] = 1
            else:
                assetclass_flag[assetclass] = 0

        for key in assetclass_cnt.keys():
            if not assetclass_cnt[key] == 1:
                sys.exit('Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(
                    key) + '. GEM strategy requires exactly one asset for this category.')

        # calculate the score
        mom12month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            mom12month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-12])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom12month': mom12month,
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})

        momentum_df['asset_weight'] = 0
        if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'mom12month'].values[0] > \
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'money_market', 'mom12month'].values[0]:
            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'mom12month'].values[0] > \
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'mom12month'].values[0]:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 1
            else:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 1
        else:
            momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 1

        return momentum_df['asset_weight'].to_list()

    def rot_weights(self):
        assetclass_allocation = {
            "gold": 0,
            "commodity": 0,
            "equity": 0,
            "equity_intl": 0,
            "bond_lt": 0,
            "bond_it": 0,
            "money_market": 0,
            "real_estate": 0
        }

        strat = {
            0: "bond_lt",
            1: "equity",
            2: "gold"
        }
        rot_indassetsclose = self.indassetsclose[0:3]
        which_max = rot_indassetsclose.index(max(rot_indassetsclose))

        winningAsset = strat.get(which_max)

        tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]

        # asset class replacements for this strategy
        # equity_intl --> equity
        # cryptocurrencies --> gold
        # tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]

        for key in assetclass_allocation:
            if key == winningAsset:
                assetclass_allocation[key] = 1.0

        assetclass_cnt = {}

        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        weights = [float(x) / y for x, y in zip(a, b)]
        return weights

    def specific_rot_weights(self):
        shares = ['TLT', 'GLD', 'VFINX']

        strat = {
            0: "bond_lt",
            1: "equity",
            2: "gold"
        }
        rot_indassetsclose = self.indassetsclose[0:3]
        which_max = rot_indassetsclose.index(max(rot_indassetsclose))

        winningAsset = strat.get(which_max)

        shares_df = pd.DataFrame(list(zip(self.params.shares, self.params.shareclass)),
                                 columns=['shares', 'shareclass'])
        shares_df = shares_df.loc[~shares_df.shareclass.isin(['non-tradable', 'benchmark', 'loan'])]
        shares_df['weights'] = 0

        if winningAsset == 'equity':
            shares_df.loc[shares_df['shares'] == 'VFINX','weights'] = 1
        elif winningAsset == 'bond_lt':
            shares_df.loc[shares_df['shares'] == 'TLT','weights'] = 1
        elif winningAsset == 'gold':
            shares_df.loc[shares_df['shares'] == 'GLD','weights'] = 1

        # asset class replacements for this strategy
        # equity_intl --> equity
        # cryptocurrencies --> gold
        # tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        weights = shares_df['weights'].tolist()
        return weights

    def specific_rot_weights2(self):
        shares = ['TLT', 'GLD', 'VFINX', 'TIP', 'QQQ', 'GSG']

        strat = {
            0: "bond_lt",
            1: "equity",
            2: "gold"
        }
        rot_indassetsclose = self.indassetsclose[0:3]
        which_max = rot_indassetsclose.index(max(rot_indassetsclose))

        winningAsset = strat.get(which_max)

        shares_df = pd.DataFrame(list(zip(self.params.shares, self.params.shareclass)),
                                 columns=['shares', 'shareclass'])
        shares_df = shares_df.loc[~shares_df.shareclass.isin(['non-tradable', 'benchmark', 'loan'])]
        shares_df['weights'] = 0

        if winningAsset == 'equity':
            shares_df.loc[shares_df['shares'] == 'VFINX','weights'] = 0.5
            shares_df.loc[shares_df['shares'] == 'QQQ','weights'] = 0.5
        elif winningAsset == 'bond_lt':
            shares_df.loc[shares_df['shares'] == 'TLT','weights'] = 1
        elif winningAsset == 'gold':
            shares_df.loc[shares_df['shares'] == 'GLD','weights'] = 0.60
            shares_df.loc[shares_df['shares'] == 'GSG','weights'] = 0.15
            shares_df.loc[shares_df['shares'] == 'TIP','weights'] = 0.25


        # asset class replacements for this strategy
        # equity_intl --> equity
        # cryptocurrencies --> gold
        # tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        weights = shares_df['weights'].tolist()
        return weights

    def onlystocks_weights(self):
        assetclass_allocation = {
            "gold": 0,
            "commodity": 0,
            "equity": 1,
            "bond_lt": 0,
            "bond_it": 0
        }

        tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable','benchmark','loan']]

        # asset class replacements for this strategy
        # equity_intl --> equity
        # money_market --> bond_it
        # cryptocurrencies --> gold
        # real_estate --> equity
        # tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
        tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
        tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

        assetclass_cnt = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        a = [0 if x is None else x for x in a]
        b = list(map(assetclass_cnt.get, tradable_shareclass))
        b = [1 if x is None else x for x in b]

        weights = [float(x) / y for x, y in zip(a, b)]
        return weights

    def specific_weights(self):
        # Required assets to execute the strategy
        shares = ['BRK-B']
        shares_df = pd.DataFrame(list(zip(self.params.shares, self.params.shareclass)),
                                              columns=['shares', 'shareclass'])
        shares_df = shares_df.loc[~shares_df.shareclass.isin(['non-tradable', 'benchmark', 'loan'])]
        shares_df['weights'] = np.where(shares_df['shares'].isin(shares), 1, 0)

        weights = shares_df['weights'].tolist()
        return weights

    def specific_fabio_adm2_weights(self):
        # Required asset classes to execute the strategy
        shares = ['SPY','QQQ','MDY','EFA','FXI','FEMKX','VNQ','GLD','TLT','LQD','VBMFX']
        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'shares':self.params.shares[0:self.params.n_assets],
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})
        momentum_df['offensive'] = np.where(momentum_df['shares']!='VBMFX', 1, 0)

        momentum_df['score'] = (momentum_df['mom3month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['shares'] == 'VBMFX', 'score'].values[0] < 0:
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 1
        else:
            idx_max_score = momentum_df.sort_values(by='score', ascending=False)['idx'][0:4]
            momentum_df.loc[idx_max_score,'asset_weight'] = 0.25

        return momentum_df['asset_weight'].to_list()

    def specific_fabiofg_adm2_weights(self):
        # Required asset classes to execute the strategy
        shares = ['SPY','QQQ','MDY','EFA','FXI','FEMKX','VNQ','GLD','TLT','LQD','VBMFX', 'IEF', 'GSG']
        adm_indassetsclose = self.indassetsclose[3]

        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'shares':self.params.shares[0:self.params.n_assets],
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})
        momentum_df['offensive'] = np.where(momentum_df['shares']!='VBMFX', 1, 0)

        momentum_df['score'] = (momentum_df['mom3month'])
        momentum_df['asset_weight'] = 0

        if momentum_df.loc[momentum_df['shares'] == 'VBMFX', 'score'].values[0] < 0:
            # momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 1
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7
        else:
            idx_max_score = momentum_df.sort_values(by='score', ascending=False)['idx'][0:4]
            momentum_df.loc[idx_max_score,'asset_weight'] = 0.25

        return momentum_df['asset_weight'].to_list()

    def specific_fabio_adm3_weights(self):
        # Required asset classes to execute the strategy
        shares = ['MDY', 'EFA', 'FEMKX', 'TLT']
        # calculate the score
        mom1month = []
        mom3month = []
        mom6month = []

        # get end of month dates for rebalancing
        dates = []
        for date_str in self.datas[0].datetime.array:
            date = bt.utils.date.num2date(date_str)
            if date.date() <= self.datas[0].datetime.date(0):
                dates.append(date)

        dates_df = pd.DataFrame(dates, columns=["date"])
        months = pd.DatetimeIndex(dates_df["date"]).month
        year = pd.DatetimeIndex(dates_df["date"]).year

        dates_df['month'] = months
        dates_df['year'] = year
        dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
        dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
        dates_eom_df = dates_eom_df.reset_index()
        month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
        dates_eom_df = dates_eom_df[month_filter]['date']

        for asset in range(0, self.params.n_assets):
            mom1month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
            mom3month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
            mom6month.append(
                self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

        momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                    'mom1month': mom1month,
                                    'mom3month': mom3month,
                                    'mom6month': mom6month,
                                    'shares':self.params.shares[0:self.params.n_assets],
                                    'tradable_shareclass': [x for x in self.params.shareclass if
                                                            x not in ['non-tradable', 'benchmark', 'loan']]})
        momentum_df['score'] = (momentum_df['mom3month'])
        momentum_df['asset_weight'] = 0

        momentum_df_assets = momentum_df[momentum_df['shares'].isin(shares)]

        shares_max_score = momentum_df_assets.sort_values(by='score', ascending=False)['shares'][0:1]
        momentum_df.loc[momentum_df['shares']==shares_max_score.values[0], 'asset_weight'] = 1

        return momentum_df['asset_weight'].to_list()

    # TODO NEW FABIO'S ALPHA
    def idiosyncratic_mom_weights(self):
        """
        Method that contains the trading logic of idiosyncratic
        """
        # initiate an empty  dictionary
        score_offensive = {}

        # define offensive assets
        assets = ['SPY','QQQ','MDY','EFA','FXI','FEMKX','VNQ','GLD','TLT','LQD']
        data = self.params.data_df[assets]
        day = self.datas[0].datetime.date(0)
        # get the idiosyncratic momentum of the different assets

        mom = utils.get_idiosyncratic_mom(assets, data, day, self.trading_days)
        # ret = get_returns(assets, self.dic_assets, day)
        # vol = get_volatility(assets, self.dic_assets, day)

        # define dict for the returns of offensive assets
        for asset in assets:
            score_offensive[asset] = mom[asset]['m3']

        asset_to_hold = sorted(score_offensive, key = score_offensive.get, reverse=True)[:1]

        shares_df = pd.DataFrame(list(zip(self.params.data_df.columns, self.params.shareclass)),
                                 columns=['shares', 'shareclass'])
        shares_df = shares_df.loc[~shares_df.shareclass.isin(['non-tradable', 'benchmark', 'loan'])]
        shares_df['asset_weight'] = np.where(shares_df["shares"].isin(asset_to_hold), 1, 0)
        return shares_df['asset_weight'].tolist()

"""
The child classes below are specific to one strategy.
"""

class customweights(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "customweights"
        super().__init__()


    def prenext(self):
        self.weights = self.params.assetweights

    def next_open(self):
        if self.rebalance_flg():
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class sixtyforty(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "sixtyforty"
        super().__init__()


    def prenext(self):
        assetclass_allocation = {
            "gold": 0,
            "commodity": 0,
            "equity": 0.6,
            "bond_lt": 0.2,
            "bond_it": 0.2
        }

        tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable','benchmark', 'loan']]

        # asset class replacements for this strategy
        # equity_intl --> equity
        # money_market --> bond_it
        # cryptocurrencies --> gold
        # real_estate --> equity
        tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
        tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
        tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]


        assetclass_cnt = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if self.rebalance_flg():
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class onlystocks(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "onlystocks"
        super().__init__()


    def prenext(self):
        assetclass_allocation = {
            "gold": 0,
            "commodity": 0,
            "equity": 1,
            "bond_lt": 0,
            "bond_it": 0
        }

        tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable','benchmark','loan']]

        # asset class replacements for this strategy
        # equity_intl --> equity
        # money_market --> bond_it
        # cryptocurrencies --> gold
        # real_estate --> equity
        tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
        tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
        tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

        assetclass_cnt = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if self.rebalance_flg():
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
                    self.broker.get_cash(),
                    self.broker.get_value(),
                    self.broker.get_fundshares(),
                    self.broker.get_fundvalue()))

            self.rebalance()

class benchmark(StandaloneStrat):
    """
    dummy class to buy the benchmark
    """
    def __init__(self):
        self.strategy_name = "benchmark"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
                    self.broker.get_cash(),
                    self.broker.get_value(),
                    self.broker.get_fundshares(),
                    self.broker.get_fundvalue()))
            self.buybenchmark()

class vanillariskparity(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "vanillariskparity"
        super().__init__()

    def prenext(self):
        assetclass_allocation = {
            "gold": 0.12,
            "commodity": 0.13,
            "equity": 0.2,
            "bond_lt": 0.15,
            "bond_it": 0.4
        }

        tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable','benchmark', 'loan']]

        # asset class replacements for this strategy
        # equity_intl --> equity
        # money_market --> bond_it
        # cryptocurrencies --> gold
        # real_estate --> equity
        tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
        tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
        tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]


        assetclass_cnt = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count

        a = list(map(assetclass_allocation.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if self.rebalance_flg():
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
                    self.broker.get_cash(),
                    self.broker.get_value(),
                    self.broker.get_fundshares(),
                    self.broker.get_fundvalue()))

            self.rebalance()

class uniform(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "uniform"
        super().__init__()

    def prenext(self):
        assetclass_allocation = {
            "gold": 0.2,
            "commodity": 0.2,
            "equity": 0.2,
            "bond_lt": 0.2,
            "bond_it": 0.2
        }

        tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable','benchmark', 'loan']]

        # asset class replacements for this strategy
        # equity_intl --> equity
        # money_market --> bond_it
        # cryptocurrencies --> gold
        # real_estate --> equity
        tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
        tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
        tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
        tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

        assetclass_cnt = {}
        assetclass_flag = {}
        for key in assetclass_allocation:
            count = sum(map(lambda x: x == key, tradable_shareclass))
            assetclass_cnt[key] = count
            if count > 0:
                assetclass_flag[key] = 1
            else:
                assetclass_flag[key] = 0

        num_assetclasses = sum(assetclass_flag.values())
        assetclass_weight = {k: v / num_assetclasses for k, v in assetclass_flag.items()}

        a = list(map(assetclass_weight.get, tradable_shareclass))
        b = list(map(assetclass_cnt.get, tradable_shareclass))

        self.weights = [float(x) / y for x, y in zip(a, b)]

    def next_open(self):
        if self.rebalance_flg():
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class rotationstrat(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "rotationstrat"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.rot_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

# Risk parity portfolio. The implementation is based on:
# https: // thequantmba.wordpress.com / 2016 / 12 / 14 / risk - parityrisk - budgeting - portfolio - in -python /
# Here the risk parity is run only at portfolio level
class riskparity(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "riskparity"
        super().__init__()


    def next_open(self):
        if self.rebalance_flg():
            target_risk = [1 / self.params.n_assets] * self.params.n_assets  # Same risk for each asset = risk parity
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in self.assetclose]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            self.weights = target_risk_contribution(target_risk, cov)

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

# Risk parity portfolio. The implementation is based on:
# https: // thequantmba.wordpress.com / 2016 / 12 / 14 / risk - parityrisk - budgeting - portfolio - in -python /
# Here the risk parity is run first at asset class level and then at portfolio level. To be used when more than an asset
# is present in each category
class riskparity_nested(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "riskparity_nested"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.riskparity_nested_weights()

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
                self.broker.get_cash(),
                self.broker.get_value(),
                self.broker.get_fundshares(),
                self.broker.get_fundvalue()))

            self.rebalance()

# Risk parity portfolio. The implementation is based on the Python library riskparity portfolio
class riskparity_pylib(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "riskparity_pylib"
        super().__init__()


    def next_open(self):
        if self.rebalance_flg():
            target_risk = [1 / self.params.n_assets] * self.params.n_assets  # Same risk for each asset = risk parity
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in self.assetclose]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            self.weights = rp.RiskParityPortfolio(covariance=cov, budget=target_risk).weights

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

# Optimal tangent portfolio according to the Modern Portfolio theory by Markowitz. The implementation is based on:
# https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/
class meanvarStrat(StandaloneStrat):
    strategy_name = "Tangent Portfolio"


    def next(self):
        print("Work in progress")

# Semi-passive strategies
class trend_u(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "trend_u"
        super().__init__()

    '''
    First allocate to each asset class and instrument; then check the trending value.
    If the current price is lower than the moving average then set the weight at 0 (that is keep cash). 
    This is a simplification of buying T-Bills. 
    '''
    def next_open(self):
        assetclass_allocation = {
            "gold": 0.2,
            "commodity": 0.2,
            "equity": 0.2,
            "bond_lt": 0.2,
            "bond_it": 0.2
        }

        if self.rebalance_flg():
            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            assetclass_cnt = {}
            assetclass_flag = {}
            for key in assetclass_allocation:
                count = sum(map(lambda x: x == key, tradable_shareclass))
                assetclass_cnt[key] = count
                if count > 0:
                    assetclass_flag[key] = 1
                else:
                    assetclass_flag[key] = 0

            num_assetclasses = sum(assetclass_flag.values())
            assetclass_weight = {k: v / num_assetclasses for k, v in assetclass_flag.items()}

            a = list(map(assetclass_weight.get, tradable_shareclass))
            b = list(map(assetclass_cnt.get, tradable_shareclass))

            self.weights = [float(x) / y for x, y in zip(a, b)]

            # Apply the trend filter here
            for asset in range(0, self.params.n_assets):
                if self.assetclose[asset].get(0)[0] < self.sma[asset][0]:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class trend2_u(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "trend2_u"
        super().__init__()

    strategy_name = "Trend Uniform"
    '''
    First allocate to each asset class and instrument; then check the trending value.
    If the current price is lower than the moving average then set the weight at 0 (that is keep cash). 
    This is a simplification of buying T-Bills. 
    '''
    def next_open(self):

        if self.rebalance_flg():
            # Apply the relative momentum filter here
            trend = []
            price = []
            for asset in range(0, self.params.n_assets):
                trend.append(self.sma[asset][0])
                price.append(self.assetclose[asset].get(0)[0])

            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            trend_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'trend': trend,
                                        'price': price,
                                        'tradable_shareclass': tradable_shareclass})
            trend_df['trend_buy_flag'] = \
                trend_df['trend'] > trend_df['price']

            tradable_shareclass_trend = list(
                trend_df.loc[trend_df['trend_buy_flag'] == True]['tradable_shareclass'].unique())
            num_assetclasses = len(tradable_shareclass_trend)

            assetclass_cnt = \
                trend_df.loc[trend_df['trend_buy_flag'] == True].groupby('tradable_shareclass').agg(['count'])[
                    'idx']
            trend_df = pd.merge(trend_df, assetclass_cnt, how="left", on='tradable_shareclass')

            conditions = [(trend_df['tradable_shareclass'].isin(tradable_shareclass_trend)) & (
                        trend_df['trend_buy_flag'] == True)]

            if num_assetclasses > 0:
                values = [1 / num_assetclasses]
            else:
                values = [0]
            trend_df['assetclass_weight'] = np.select(conditions, values)
            trend_df['asset_weight'] = trend_df['assetclass_weight'] / trend_df['count']
            trend_df['asset_weight'] = trend_df['asset_weight'].replace(np.nan, 0)

            self.weights = trend_df['asset_weight'].to_list()

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class trend_rp(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "trend_rp"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            target_risk = [1 / self.params.n_assets] * self.params.n_assets  # Same risk for each asset = risk parity
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in self.assetclose]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            self.weights = target_risk_contribution(target_risk, cov)

            # Apply the trend filter here
            for asset in range(0, self.params.n_assets):
                if self.assetclose[asset].get(0)[0] < self.sma[asset][0]:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class absmom_u(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "absmom_u"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            assetclass_allocation = {
                "gold": 0.2,
                "commodity": 0.2,
                "equity": 0.2,
                "bond_lt": 0.2,
                "bond_it": 0.2
            }

            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            assetclass_cnt = {}
            assetclass_flag = {}
            for key in assetclass_allocation:
                count = sum(map(lambda x: x == key, tradable_shareclass))
                assetclass_cnt[key] = count
                if count > 0:
                    assetclass_flag[key] = 1
                else:
                    assetclass_flag[key] = 0

            num_assetclasses = sum(assetclass_flag.values())
            assetclass_weight = {k: v / num_assetclasses for k, v in assetclass_flag.items()}

            a = list(map(assetclass_weight.get, tradable_shareclass))
            b = list(map(assetclass_cnt.get, tradable_shareclass))

            self.weights = [float(x) / y for x, y in zip(a, b)]

            # Apply the absolute momentum filter here
            for asset in range(0, self.params.n_assets):
                if self.momentum[asset][0] < 0:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class absmom_rp(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "absmom_rp"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            target_risk = [1 / self.params.n_assets] * self.params.n_assets  # Same risk for each asset = risk parity
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in self.assetclose]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            self.weights = target_risk_contribution(target_risk, cov)

            # Apply the absolute momentum filter here
            for asset in range(0, self.params.n_assets):
                if self.momentum[asset][0] < 0:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class relmom_u(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "relmom_u"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            # Apply the relative momentum filter here
            momentum = []
            for asset in range(0, self.params.n_assets):
                momentum.append(self.momentum[asset][0])

            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'momentum': momentum,
                                        'tradable_shareclass': tradable_shareclass})
            momentum_df['momentum_buy_flag'] = \
                momentum_df['momentum'] > momentum_df['momentum'].quantile(self.params.momentum_percentile)

            tradable_shareclass_momentum = list(
                momentum_df.loc[momentum_df['momentum_buy_flag'] == True]['tradable_shareclass'].unique())
            num_assetclasses = len(tradable_shareclass_momentum)

            assetclass_cnt = \
                momentum_df.loc[momentum_df['momentum_buy_flag'] == True].groupby('tradable_shareclass').agg(['count'])[
                    'idx']
            momentum_df = pd.merge(momentum_df, assetclass_cnt, how="left", on='tradable_shareclass')

            conditions = [(momentum_df['tradable_shareclass'].isin(tradable_shareclass_momentum)) & (
                        momentum_df['momentum_buy_flag'] == True)]
            values = [1 / num_assetclasses]
            momentum_df['assetclass_weight'] = np.select(conditions, values)
            momentum_df['asset_weight'] = momentum_df['assetclass_weight'] / momentum_df['count']
            momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)

            self.weights = momentum_df['asset_weight'].to_list()

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class relmom_rp(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "relmom_rp"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            momentum = []
            for asset in range(0, self.params.n_assets):
                momentum.append(self.momentum[asset][0])

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'momentum': momentum,
                                        'tradable_shareclass': [x for x in self.params.shareclass if
                                                                x not in ['non-tradable', 'benchmark', 'loan']]})
            momentum_df['momentum_buy_flag'] = momentum_df['momentum'] > momentum_df['momentum'].quantile(self.params.momentum_percentile)
            n_momentum_assets = momentum_df['momentum_buy_flag'].sum()

            target_risk = [1 / n_momentum_assets] * n_momentum_assets # Same risk for each asset in momentum portfolio = risk parity
            assetclose_momentum = [self.assetclose[i] for i in range(0, self.params.n_assets) if momentum_df['momentum_buy_flag'][i]]
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in assetclose_momentum]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            weights_momentum = target_risk_contribution(target_risk, cov)
            momentum_idx = momentum_df.loc[momentum_df['momentum_buy_flag']]['idx'].to_list()
            weights_df = pd.DataFrame({'idx':momentum_idx, 'asset_weight':weights_momentum})
            momentum_df = pd.merge(left=momentum_df, right=weights_df, how='left', on='idx')
            momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)

            self.weights = momentum_df['asset_weight'].to_list()

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class momtrend_u(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "momtrend_u"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            # Apply the relative momentum filter here
            momentum = []
            for asset in range(0, self.params.n_assets):
                momentum.append(self.momentum[asset][0])

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'momentum': momentum,
                                        'tradable_shareclass': tradable_shareclass})
            momentum_df['momentum_buy_flag'] = momentum_df['momentum'] > momentum_df['momentum'].quantile(self.params.momentum_percentile)

            tradable_shareclass_momentum = list(
                momentum_df.loc[momentum_df['momentum_buy_flag'] == True]['tradable_shareclass'].unique())
            num_assetclasses = len(tradable_shareclass_momentum)

            assetclass_cnt = \
                momentum_df.loc[momentum_df['momentum_buy_flag'] == True].groupby('tradable_shareclass').agg(['count'])[
                    'idx']
            momentum_df = pd.merge(momentum_df, assetclass_cnt, how="left", on='tradable_shareclass')

            conditions = [(momentum_df['tradable_shareclass'].isin(tradable_shareclass_momentum)) & (
                        momentum_df['momentum_buy_flag'] == True)]
            values = [1 / num_assetclasses]
            momentum_df['assetclass_weight'] = np.select(conditions, values)
            momentum_df['asset_weight'] = momentum_df['assetclass_weight'] / momentum_df['count']
            momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)

            self.weights = momentum_df['asset_weight'].to_list()

            # Apply the trend filter here
            for asset in range(0, self.params.n_assets):
                if self.assetclose[asset].get(0)[0] < self.sma[asset][0]:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class momtrend_rp(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "momtrend_rp"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            momentum = []
            for asset in range(0, self.params.n_assets):
                momentum.append(self.momentum[asset][0])

            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'momentum': momentum,
                                        'tradable_shareclass': tradable_shareclass})
            momentum_df['momentum_buy_flag'] = momentum_df['momentum'] > momentum_df['momentum'].quantile(self.params.momentum_percentile)
            n_momentum_assets = momentum_df['momentum_buy_flag'].sum()

            target_risk = [1 / n_momentum_assets] * n_momentum_assets # Same risk for each asset in momentum portfolio = risk parity
            assetclose_momentum = [self.assetclose[i] for i in range(0, self.params.n_assets) if momentum_df['momentum_buy_flag'][i]]
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in assetclose_momentum]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            weights_momentum = target_risk_contribution(target_risk, cov)
            momentum_idx = momentum_df.loc[momentum_df['momentum_buy_flag']]['idx'].to_list()
            weights_df = pd.DataFrame({'idx':momentum_idx, 'asset_weight':weights_momentum})
            momentum_df = pd.merge(left=momentum_df, right=weights_df, how='left', on='idx')
            momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)

            self.weights = momentum_df['asset_weight'].to_list()

            # Apply the trend filter here
            for asset in range(0, self.params.n_assets):
                if self.assetclose[asset].get(0)[0] < self.sma[asset][0]:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class momtrelabs_u(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "momtrelabs_u"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            # Apply the relative momentum filter here
            momentum = []
            for asset in range(0, self.params.n_assets):
                momentum.append(self.momentum[asset][0])

            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'momentum': momentum,
                                        'tradable_shareclass': tradable_shareclass})
            momentum_df['momentum_buy_flag'] = momentum_df['momentum'] > momentum_df['momentum'].quantile(self.params.momentum_percentile)

            tradable_shareclass_momentum = list(
                momentum_df.loc[momentum_df['momentum_buy_flag'] == True]['tradable_shareclass'].unique())
            num_assetclasses = len(tradable_shareclass_momentum)

            assetclass_cnt = \
                momentum_df.loc[momentum_df['momentum_buy_flag'] == True].groupby('tradable_shareclass').agg(['count'])[
                    'idx']
            momentum_df = pd.merge(momentum_df, assetclass_cnt, how="left", on='tradable_shareclass')

            conditions = [(momentum_df['tradable_shareclass'].isin(tradable_shareclass_momentum)) & (
                        momentum_df['momentum_buy_flag'] == True)]
            values = [1 / num_assetclasses]
            momentum_df['assetclass_weight'] = np.select(conditions, values)
            momentum_df['asset_weight'] = momentum_df['assetclass_weight'] / momentum_df['count']
            momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)

            self.weights = momentum_df['asset_weight'].to_list()

            # Apply the absolute momentum filter here
            for asset in range(0, self.params.n_assets):
                if self.momentum[asset][0] < 0:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class momrelabs_rp(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "momrelabs_rp"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            momentum = []
            for asset in range(0, self.params.n_assets):
                momentum.append(self.momentum[asset][0])

            tradable_shareclass = [x for x in self.params.shareclass if x not in ['non-tradable', 'benchmark', 'loan']]
            # asset class replacements for this strategy
            # equity_intl --> equity
            # money_market --> bond_it
            # cryptocurrencies --> gold
            # real_estate --> equity
            tradable_shareclass = ["equity" if x == "equity_intl" else x for x in tradable_shareclass]
            tradable_shareclass = ["bond_it" if x == "money_market" else x for x in tradable_shareclass]
            tradable_shareclass = ["gold" if x == "cryptocurrencies" else x for x in tradable_shareclass]
            tradable_shareclass = ["equity" if x == "real_estate" else x for x in tradable_shareclass]

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'momentum': momentum,
                                        'tradable_shareclass': tradable_shareclass})
            momentum_df['momentum_buy_flag'] = momentum_df['momentum'] > momentum_df['momentum'].quantile(self.params.momentum_percentile)
            n_momentum_assets = momentum_df['momentum_buy_flag'].sum()

            target_risk = [1 / n_momentum_assets] * n_momentum_assets # Same risk for each asset in momentum portfolio = risk parity
            assetclose_momentum = [self.assetclose[i] for i in range(0, self.params.n_assets) if momentum_df['momentum_buy_flag'][i]]
            thisAssetClose = [x.get(size=self.params.lookback_period_long) for x in assetclose_momentum]
            # Check if asset prices is equal to the lookback period for all assets exist
            thisAssetClose_len = [len(i) for i in thisAssetClose]
            if not all(elem == self.params.lookback_period_long for elem in thisAssetClose_len):
                return

            logrets = [np.diff(np.log(x)) for x in thisAssetClose]

            if self.params.corrmethod == 'pearson':
                corr = np.corrcoef(logrets)
            elif self.params.corrmethod == 'spearman':
                corr, p, = stats.spearmanr(logrets, axis=1)

            stddev = np.array([np.std(x[len(x) - self.params.lookback_period_short:len(x)]) for x in logrets])  # standard dev indicator
            stddev_matrix = np.diag(stddev)
            cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

            weights_momentum = target_risk_contribution(target_risk, cov)
            momentum_idx = momentum_df.loc[momentum_df['momentum_buy_flag']]['idx'].to_list()
            weights_df = pd.DataFrame({'idx':momentum_idx, 'asset_weight':weights_momentum})
            momentum_df = pd.merge(left=momentum_df, right=weights_df, how='left', on='idx')
            momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)

            self.weights = momentum_df['asset_weight'].to_list()

            # Apply the absolute momentum filter here
            for asset in range(0, self.params.n_assets):
                if self.momentum[asset][0] < 0:
                    self.weights[asset] = 0

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class GEM(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "GEM"
        super().__init__()

    """
    https://blog.thinknewfound.com/2019/01/fragility-case-study-dual-momentum-gem/
    """

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.gem_weights()

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class acc_dualmom(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "ADM"
        super().__init__()

    """
    https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/
    """

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.adm_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class acc_dualmom2(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "ADM2"
        super().__init__()

    """
    https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/
    """

    def next_open(self):
        if self.rebalance_flg():
            # Required asset classes to execute the strategy
            assetclasses = ["equity", "equity_intl", "bond_lt", "gold", "commodity"]

            # Get the assets belonging to the categories required and ignore the rest
            strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]

            # Check that only 3 assets are delivered with the shareclasses defined above
            assetclass_cnt = {}
            assetclass_flag = {}
            for assetclass in assetclasses:
                count = sum(map(lambda x: x == assetclass, strat_shareclass))
                assetclass_cnt[assetclass] = count
                if count > 0:
                    assetclass_flag[assetclass] = 1
                else:
                    assetclass_flag[assetclass] = 0

            for key in assetclass_cnt.keys():
                if not assetclass_cnt[key] == 1:
                    sys.exit(
                        'Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(key) +
                        '. Accelerating dual moment strategy requires exactly one asset for this category.')

            adm_indassetsclose = self.indassetsclose[3]

            # calculate the score
            mom1month = []
            mom3month = []
            mom6month = []

            # get end of month dates for rebalancing
            dates = []
            for date_str in self.datas[0].datetime.array:
                date = bt.utils.date.num2date(date_str)
                if date.date() <= self.datas[0].datetime.date(0):
                    dates.append(date)

            dates_df = pd.DataFrame(dates, columns=["date"])
            months = pd.DatetimeIndex(dates_df["date"]).month
            year = pd.DatetimeIndex(dates_df["date"]).year

            dates_df['month'] = months
            dates_df['year'] = year
            dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
            dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
            dates_eom_df = dates_eom_df.reset_index()
            month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
            dates_eom_df = dates_eom_df[month_filter]['date']

            for asset in range(0, self.params.n_assets):
                # mom1month.append(self.mom1month[asset][0])
                # mom3month.append(self.mom3month[asset][0])
                # mom6month.append(self.mom6month[asset][0])
                mom1month.append(self.assetclose[asset][0] / self.assetclose[asset][
                    -sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
                mom3month.append(self.assetclose[asset][0] / self.assetclose[asset][
                    -sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
                mom6month.append(self.assetclose[asset][0] / self.assetclose[asset][
                    -sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'mom1month': mom1month,
                                        'mom3month': mom3month,
                                        'mom6month': mom6month,
                                        'tradable_shareclass': [x for x in self.params.shareclass if
                                                                x not in ['non-tradable', 'benchmark',
                                                                          'loan']]})

            momentum_df['score'] = (
                        momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month'])
            momentum_df['asset_weight'] = 0

            if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > \
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0]:
                if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'score'].values[0] > \
                        adm_indassetsclose[0]:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity', 'asset_weight'] = 1
                else:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.4
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.2
            else:
                if momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'score'].values[0] > \
                        adm_indassetsclose[0]:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 1
                else:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'asset_weight'] = 0.4
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 0.4
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'commodity', 'asset_weight'] = 0.2

            self.weights = momentum_df['asset_weight'].to_list()

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
                self.broker.get_cash(),
                self.broker.get_value(),
                self.broker.get_fundshares(),
                self.broker.get_fundvalue()))

            self.rebalance()

class rot_adm(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "rot_adm"
        super().__init__()

    """
    rotation strategy + accelerating dual momentum
    https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/
    """

    def next_open(self):
        if self.rebalance_flg():
            # Required asset classes to execute the strategy
            assetclasses = ["equity","equity_intl","bond_lt","gold"]

            # Get the assets belonging to the categories required and ignore the rest
            strat_shareclass = [x for x in self.params.shareclass if x in assetclasses]

            # strat_shareclass = []
            # for x in self.params.shareclass:
            #     if x in assetclasses:
            #         strat_shareclass.append(x)

            # Check that only 4 assets are delivered with the shareclasses defined above
            assetclass_cnt = {}
            assetclass_flag = {}
            for assetclass in assetclasses:
                count = sum(map(lambda x: x == assetclass, strat_shareclass))
                assetclass_cnt[assetclass] = count
                if count > 0:
                    assetclass_flag[assetclass] = 1
                else:
                    assetclass_flag[assetclass] = 0

            for key in assetclass_cnt.keys():
                if not assetclass_cnt[key] == 1:
                    sys.exit('Error: ' + str(assetclass_cnt[key]) + ' assets found for the category ' + str(key) +
                             '. Rotation + Accelerating dual momentum strategy requires exactly one asset for this category.')

            # calculate the score
            mom1month = []
            mom3month = []
            mom6month = []

            # get end of month dates for rebalancing
            dates = []
            for date_str in self.datas[0].datetime.array:
                date = bt.utils.date.num2date(date_str)
                if date.date() <= self.datas[0].datetime.date(0):
                    dates.append(date)

            dates_df = pd.DataFrame(dates, columns=["date"])
            months = pd.DatetimeIndex(dates_df["date"]).month
            year = pd.DatetimeIndex(dates_df["date"]).year

            dates_df['month'] = months
            dates_df['year'] = year
            dates_eom_df = dates_df.groupby(['month', 'year']).max().sort_values('date')
            dates_eom_df.drop(dates_eom_df.tail(1).index, inplace=True)
            dates_eom_df = dates_eom_df.reset_index()
            month_filter = pd.Series(range(0, len(dates_eom_df))) % self.params.reb_period == 0
            dates_eom_df = dates_eom_df[month_filter]['date']

            for asset in range(0, self.params.n_assets):
                # mom1month.append(self.mom1month[asset][0])
                # mom3month.append(self.mom3month[asset][0])
                # mom6month.append(self.mom6month[asset][0])
                mom1month.append(self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-1])] - 1)
                mom3month.append(self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-3])] - 1)
                mom6month.append(self.assetclose[asset][0] / self.assetclose[asset][-sum(dates_df['date'] > dates_eom_df.iloc[-6])] - 1)
            # tradable_shareclass = []
            # for x in self.params.shareclass:
            #     if x not in ['non-tradable', 'benchmark']:
            #         tradable_shareclass.append(x)
            #
            # momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
            #                             'mom1month': mom1month,
            #                             'mom3month': mom3month,
            #                             'mom6month': mom6month,
            #                             'tradable_shareclass':tradable_shareclass })

            momentum_df = pd.DataFrame({'idx': range(0, self.params.n_assets),
                                        'mom1month': mom1month,
                                        'mom3month': mom3month,
                                        'mom6month': mom6month,
                                        'tradable_shareclass': [x for x in self.params.shareclass if
                                                                x not in ['non-tradable', 'benchmark', 'loan']]})

            momentum_df['score'] = momentum_df['mom1month'] + momentum_df['mom3month'] + momentum_df['mom6month']
            adm_indassetsclose = self.indassetsclose[3]
            # adm without gold
            momentum_df['adm_weight'] = 0
            if momentum_df.loc[momentum_df['tradable_shareclass']=='equity','score'].values[0] > momentum_df.loc[momentum_df['tradable_shareclass']=='equity_intl','score'].values[0]:
                if momentum_df.loc[momentum_df['tradable_shareclass']=='equity','score'].values[0] > adm_indassetsclose[0]:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity','adm_weight'] = 1
                else:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt','adm_weight'] = 1
            else:
                if momentum_df.loc[momentum_df['tradable_shareclass']=='equity_intl','score'].values[0] > adm_indassetsclose[0]:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl','adm_weight'] = 1
                else:
                    momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt','adm_weight'] = 1

            # adm with gold
            momentum_df['adm_gold_weight'] = 0
            if momentum_df.score.iloc[momentum_df['score'].idxmax(axis=0)] > 0:
                momentum_df.adm_gold_weight.iloc[momentum_df['score'].idxmax(axis=0)] = 1
            else:
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'bond_lt', 'adm_gold_weight'] = 1

            # rotation
            strat = {
                0: "bond_lt",
                1: "equity",
                2: "gold"
            }

            rot_indassetsclose = self.indassetsclose[0:3]
            which_max = rot_indassetsclose.index(max(rot_indassetsclose))
            winningAsset = strat.get(which_max)
            momentum_df.loc[momentum_df['tradable_shareclass'] == winningAsset, 'rot_weight'] = 1
            momentum_df['rot_weight'] = momentum_df['rot_weight'].replace(np.nan, 0)

            # combination

            # if rotation strategy is in gold and gold score is max --> gold
            if (momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold','adm_gold_weight'].values[0] == 1) and \
                (momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold','rot_weight'].values[0] == 1):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'gold', 'asset_weight'] = 1
                momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)
            # if rotation strategy is in equity and equity_intl score is max --> equity_intl
            elif (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl','adm_weight'].values[0] == 1) and \
                (momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity','rot_weight'].values[0] == 1):
                momentum_df.loc[momentum_df['tradable_shareclass'] == 'equity_intl', 'asset_weight'] = 1
                momentum_df['asset_weight'] = momentum_df['asset_weight'].replace(np.nan, 0)
            # in all other cases take the average between rotation and adm
            else:
                momentum_df.loc[:, 'asset_weight'] = momentum_df.loc[:,'adm_weight']*2/4 + momentum_df.loc[:,'rot_weight'] *2/4

            self.weights = momentum_df['asset_weight'].to_list()
            #print(self.weights)
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class rot_adm_dual_rp(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "rot_adm_dual_rp"
        super().__init__()

    """
    rotation strategy + accelerating dual momentum + antonacci's dual momentum + risk parity
    https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/
    """

    def next_open(self):
        if self.rebalance_flg():
            weights_adm = self.adm_weights()
            weights_gem = self.gem_weights()
            weights_rp = self.riskparity_nested_weights()
            weights_rot = self.rot_weights()
            self.weights = ((np.array(weights_adm) + np.array(weights_gem) + np.array(weights_rp) + np.array(weights_rot))/4).tolist()

            #print(self.weights)
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class vigilant(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Vigilant"
        super().__init__()

    """
    https://allocatesmartly.com/vigilant-asset-allocation-dr-wouter-keller-jw-keuning/
    """

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_vigilant_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class fabios(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Fabio's strategies"
        super().__init__()

    """
    Fabio's strategies
    """

    def next_open(self):
        if self.rebalance_flg():
            # TEST FABIOS'
            self.weights = ((np.array(self.specific_vigilant_weights()) +
                           np.array(self.specific_rot_weights()) +
                           np.array(self.specific_adm_weights()) +
                           np.array(self.specific_fabio_adm2_weights()) +
                           np.array(self.idiosyncratic_mom_weights()) +
                           np.array(self.specific_fabio_adm3_weights()) )/6).tolist()
                           # np.array(self.specific_weights()) )/6).tolist()

            print(self.weights)
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class TESTING(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "rot_adm_dual_rp"
        super().__init__()

    """
    rotation strategy + accelerating dual momentum + antonacci's dual momentum + risk parity
    https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/
    """

    def next_open(self):
        if self.rebalance_flg():

            # TEST FABIOS'
            # weights = ((np.array(self.specific_vigilant_weights()) +
            #                np.array(self.specific_rot_weights()) +
            #                np.array(self.specific_adm_weights()) +
            #                np.array(self.specific_fabio_adm2_weights()) +
            #                np.array(self.specific_fabio_adm3_weights()) )/5).tolist()
            #                # np.array(self.specific_weights()) )/6).tolist()

            # WITH NEW fabio's alpha STRAT
            # weights = ((np.array(self.specific_vigilant_weights()) +
            #                np.array(self.specific_rot_weights()) +
            #                np.array(self.specific_adm_weights()) +
            #                np.array(self.specific_fabio_adm2_weights()) +
            #                np.array(self.idiosyncratic_mom_weights()) +
            #                np.array(self.specific_fabio_adm3_weights()) )/6).tolist()
            #                # np.array(self.specific_weights()) )/6).tolist()

            # FABIO'S WITHOUT ADM, BRK-B AND WITH ALPHA
            weights = ((np.array(self.specific_vigilant_weights()) +
                           np.array(self.specific_rot_weights()) +
                           np.array(self.specific_fabio_adm2_weights()) +
                           np.array(self.idiosyncratic_mom_weights()) +
                           np.array(self.specific_fabio_adm3_weights()) )/5).tolist()

            # with ADM FG and ROT FG
            # WITH NEW fabio's alpha STRAT
            # weights = (
            #                0.15*np.array(self.specific_vigilant_weights()) +
            #                0.15*np.array(self.specific_rot_weights2()) +
            #                # 0.18*np.array(self.specific_adm_grad_div_weights_2()) +
            #                0.15*np.array(self.specific_adm_grad_div_weights()) +
            #                0.15*np.array(self.specific_fabiofg_adm2_weights()) +
            #                0.15*np.array(self.specific_fabio_adm3_weights()) +
            #                0.15*np.array(self.idiosyncratic_mom_weights()) +
            #                0.1*np.array(self.specific_weights())).tolist()

            # WITHOUT NEW fabio's alpha STRAT
            # weights = (
            #                0.18*np.array(self.specific_vigilant_weights()) +
            #                0.18*np.array(self.specific_rot_weights2()) +
            #                0.18*np.array(self.specific_adm_grad_div_weights()) +
            #                0.18*np.array(self.specific_fabiofg_adm2_weights()) +
            #                0.18*np.array(self.specific_fabio_adm3_weights()) +
            #                0.1*np.array(self.specific_weights())).tolist()

            # weights = (
            #                0.18*np.array(self.specific_vigilant_weights()) +
            #                0.18*np.array(self.specific_rot_weights2()) +
            #                # 0.18*np.array(self.specific_adm_grad_div_weights_2()) +
            #                0.18*np.array(self.specific_adm_grad_div_weights()) +
            #                0.18*np.array(self.specific_fabiofg_adm2_weights()) +
            #                0.18*np.array(self.specific_fabio_adm3_weights()) +
            #                0.1*np.array(self.specific_weights())).tolist()


            shares_df = pd.DataFrame(list(zip(self.params.data_df.columns, self.params.shareclass)),
                                     columns=['shares', 'shareclass'])
            shares_df = shares_df.loc[~shares_df.shareclass.isin(['non-tradable', 'benchmark', 'loan'])]['shares']
            dic_assets = self.params.data_df[shares_df]
            day = self.datas[0].datetime.date(0)
            assets = shares_df.tolist()
            ann_vol = utils.get_volatility(assets, dic_assets, day, self.trading_days)

            vol_scaling = []
            for asset in assets:
                vol_scaling.append(ann_vol[asset]['m12']/ann_vol[asset]['m3'])

            weights = [a * b for a, b in zip(vol_scaling, weights)]

            # self.weights = [number / sum(weights) for number in weights]

            if sum(weights) > 1:
                self.weights = [number / sum(weights) for number in weights]
                self.leverage = self.params.leverage + (sum(weights) - 1)
            else:
                self.weights = weights
                self.leverage = self.params.leverage

            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))

            self.rebalance()

class specific_vigilant(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "specific vigilant"
        super().__init__()

    """
    https://allocatesmartly.com/vigilant-asset-allocation-dr-wouter-keller-jw-keuning/
    """

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_vigilant_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific_rot(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Specific Rotational Strategy"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_rot_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific_rot2(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Specific Rotational Strategy 2"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_rot_weights2()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific_adm(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Specific ADM"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_adm_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific_adm_grad_div(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Specific ADM gradual & diversified"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_adm_grad_div_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific_fabio_adm2(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Specific ADM Fabio 2"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_fabio_adm2_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific_fabiofg_adm2(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "specific fabio & fg adm 2"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_fabiofg_adm2_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific_fabio_adm3(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "specific fabio adm3"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_fabio_adm3_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()

class specific(StandaloneStrat):
    def __init__(self):
        self.strategy_name = "Specific asset strat"
        super().__init__()

    def next_open(self):
        if self.rebalance_flg():
            self.weights = self.specific_weights()
            self.log("Pre-rebalancing CASH %.2f, VALUE  %.2f, FUND SHARES %.2f, FUND VALUE %.2f:" % (
            self.broker.get_cash(),
            self.broker.get_value(),
            self.broker.get_fundshares(),
            self.broker.get_fundvalue()))
            self.rebalance()