import pandas as pd
import os
import backtrader as bt
from strategies import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pandas_datareader.data as web
import math
import numpy.random as nrand
from GLOBAL_VARS import *
from PortfolioDB import PortfolioDB
import platform
import streamlit as st
import base64
from pathlib import Path
import datetime
import threading
from time import time, sleep
import pandas as pd
import requests
from pprint import pprint
from scipy.stats import kurtosis, skew
from dateutil.relativedelta import relativedelta


import yfinance as yf
# yf.pdr_override()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=True)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# finds file in a folder
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files or name in dirs:
            return os.path.join(root, name)

# Converts a date in "yyyy-mm-dd" format to a dateTime object
def convertDate(dateString):
   return datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S')

# Takes in a date in the format "yyyy-mm-dd hh:mm:ss" and increments it by one day. Or if the
# day is a Friday, increment by 3 days, so the next day of data we get is the next
# Monday.
def incrementDate(dateString):
    dateTime = datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S')
    # If the day of the week is a friday increment by 3 days.
    if dateTime.isoweekday() == 5:
        datePlus = dateTime + timedelta(3)
    else:
        datePlus = dateTime + timedelta(1)
    return str(datePlus)

# Clear the output folder only the first time this is run
@st.cache(allow_output_mutation=True)
def delete_output_first():
    outputdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputdir = find("output", outputdir)
    delete_in_dir(outputdir)

def delete_in_dir(mydir, *args, **kwargs):
    """
    Deletes all files in a directory
    """

    file_extension = kwargs.get('file_extension', None)
    if file_extension is None:
        filelist = [f for f in os.listdir(mydir)]
    else:
        filelist = [f for f in os.listdir(mydir) if f.endswith(file_extension)]

    for f in filelist:
        os.remove(os.path.join(mydir, f))

def print_section_divider(strategy_name):
    """
    Prints a section divider with the strategy name
    """
    print("##############################################################################")
    print("###")
    print("### Backtest strategy: " + strategy_name)
    print("###")
    print("##############################################################################")

def print_header(params, strategy_list):
    print('##############################################################################')
    print('##############################################################################')
    print('### Backtest starting')
    print('###  Parameters:')
    print('###    historic' + ' ' + str(params['historic']))
    print('###    shares' + ' ' + str(params['shares']))
    print('###    shareclass' + ' ' + str(params['shareclass']))
    print('###    weights' + ' ' + str(params['weights']))
    print('###    initial_cash' + ' ' + str(params['initial_cash']))
    print('###    contribution' + ' ' + str(params['contribution']))
    print('###    create_report' + ' ' + str(params['create_report']))
    print('###    report_name' + ' ' + str(params['report_name']))
    print('###    strategy' + ' ' + str(strategy_list))
    print('###    startdate' + ' ' + str(params['startdate']))
    print('###    enddate' + ' ' + str(params['enddate']))
    print('###    leverage' + ' ' + str(params['leverage']))
    print('##############################################################################')

def import_histprices_db(dataLabel):
    db = PortfolioDB(databaseName = DB_NAME)
    df = db.getPrices(dataLabel)

    stock_info = db.getStockInfo(dataLabel)
    if stock_info['treatment_type'].values[0] == 'yield':
        maturity = stock_info['maturity'].values[0]
        if not type(maturity) == np.float64:
            print("Error: maturity is needed for ticker " + dataLabel + '. Please update DIM_STOCKS.')
            #return

        frequency = stock_info['frequency'].values[0]
        if frequency == 'D':
            dt = 1 / params['DAYS_IN_YEAR_BOND_PRICE']
        elif frequency == 'Y':
            dt = 1

        total_return = bond_total_return(ytm=df[['close']], dt=dt, maturity=maturity)
        df['close'] = 100 * np.exp(np.cumsum(total_return['total_return']))
        df['close'].iloc[0] = 100
        df = df.dropna()

    df = df.set_index('date')
    df.index.name = 'Date'
    df = df[['close','open','high','low','volume']]

    return df

def get_loan(startdate, enddate, BM_rate_flg, interest):
    """
    Simulates the interests paid for a margin loan taken with the broker
    The benchmark rate is th e fed funds rates. The broker interest rate is added to it.
    """
    benchmark_rate = load_fred_curve(startdate, enddate, ['FEDFUNDS'])
    if BM_rate_flg == True:
        benchmark_rate["FEDFUNDS"] = benchmark_rate["FEDFUNDS"] / 100 + interest / 100
    else:
        benchmark_rate["FEDFUNDS"] = interest / 100
    benchmark_rate["daily_rate"] = benchmark_rate["FEDFUNDS"] / 360
    dates = pd.DataFrame(index=pd.date_range(startdate, enddate))
    benchmark_rate_dates = pd.merge(dates, benchmark_rate, how="left", left_index=True, right_index=True)
    benchmark_rate_dates = benchmark_rate_dates.fillna(method='bfill')
    benchmark_rate_dates = benchmark_rate_dates.fillna(method='ffill')
    benchmark_rate_dates["close"] = 1 * (1 + benchmark_rate_dates["daily_rate"]).cumprod()
    cash_df = benchmark_rate_dates[['close']]
    return cash_df

def add_expenses(proxy, expense_ratio=0.0, timeframe=bt.TimeFrame.Days):
    """
    Add an expense ratio to an ETF.
    Daily percent change is calculated by taking the daily return of
    the price, subtracting the daily expense ratio.
    """
    initial_value = proxy.iloc[0]
    pct_change = proxy.pct_change(1)
    if timeframe == bt.TimeFrame.Days:
        pct_change = (pct_change - expense_ratio / params['DAYS_IN_YEAR'])
    elif timeframe == bt.TimeFrame.Years:
        pct_change = ((1 + pct_change) ** (1 / params['DAYS_IN_YEAR'])) - 1 # Transform into daily returns
        pct_change = (pct_change - expense_ratio / params['DAYS_IN_YEAR']) # Apply expense ratio
        pct_change = ((1 + pct_change) ** params['DAYS_IN_YEAR']) - 1 # Re-transform into yearly returns
    new_price = initial_value * (1 + pct_change).cumprod()
    new_price.iloc[0] = initial_value
    return new_price

"""
bond total return based on formula 5 of paper 
https://mpra.ub.uni-muenchen.de/92607/1/MPRA_paper_92607.pdf
See also https://quant.stackexchange.com/questions/22837/how-to-calculate-us-treasury-total-return-from-yield

"""
def bond_total_return(ytm, dt, maturity):
    ytm_pc = ytm.to_numpy()/100

    P0 = 1/np.power(1+ytm_pc, maturity) # price

    # first and second price derivatives
    P0_backward = np.roll(P0, 1)
    P0_backward = np.delete(P0_backward, 0, axis=0)
    P0_backward = np.delete(P0_backward, len(P0_backward)-1, axis=0)

    P0_forward = np.roll(P0, -1)
    P0_forward = np.delete(P0_forward, 0, axis=0)
    P0_forward = np.delete(P0_forward, len(P0_forward) - 1, axis=0)

    d_ytm = np.roll(np.diff(ytm_pc, axis=0),-1)
    d_ytm = np.delete(d_ytm, len(d_ytm)-1, axis=0)

    dP0_dytm = (P0_forward-P0_backward)/(2*d_ytm)
    dP0_dytm[dP0_dytm == np.inf] = 0
    dP0_dytm[dP0_dytm == -np.inf] = 0
    dP0_dytm[np.isnan(dP0_dytm)] = 0

    d2P0_dytm2 = (P0_forward - 2 * P0[1:len(P0)-1] + P0_backward) / (np.power(d_ytm, 2))
    d2P0_dytm2[d2P0_dytm2 == np.inf] = 0
    d2P0_dytm2[d2P0_dytm2 == -np.inf] = 0
    d2P0_dytm2[np.isnan(d2P0_dytm2)] = 0

    # Duration and convexity
    duration = -dP0_dytm/P0[1:len(P0)-1]
    convexity = d2P0_dytm2/P0[1:len(P0)-1]

    yield_income = (np.log(1+ytm_pc[1:len(ytm_pc)-1])*dt)   # First term
    duration_term = -duration * d_ytm
    convexity_term = 0.5 * (convexity - np.power(duration,2)) * np.power(d_ytm,2)
    time_term = (1/(1+ytm_pc))[1:len(ytm_pc)-1]*dt*d_ytm
    total_return = yield_income + duration_term + convexity_term + time_term

    total_return_df = pd.DataFrame(data=total_return, index=ytm.index[1:len(ytm.index)-1])
    total_return_df.columns = ["total_return"]
    return total_return_df

def common_dates(data, shareclass, fromdate, todate, timeframe):
    # Get latest startdate and earlier end date
    start = fromdate
    end = todate
    n_tradable = sum(1 for n in shareclass if n != 'non-tradable' and n != 'loan')
    for i in range(0, n_tradable):
        this_start = max(data[i].index[0].date(), fromdate)
        start = max(this_start, start)
        this_end = min(data[i].index[-1].date(), todate)
        end = min(this_end, end)

    if timeframe == bt.TimeFrame.Days: # 5
        dates = pd.bdate_range(start, end)
    elif timeframe == bt.TimeFrame.Years: # 8
        dates = pd.date_range(start,end,freq='ys')

    left = pd.DataFrame(index=dates)
    data_dates = []
    for i in range(0, len(data)):
        right=data[i]
        this_data_dates = pd.merge(left, right, left_index=True,right_index=True, how="left")
        this_data_dates = this_data_dates.fillna(method='ffill')
        data_dates.append(this_data_dates)
    return data_dates

class WeightsObserver(bt.observer.Observer):
    params = (('n_assets', 100),)  # set conservatively to 100 as the dynamic assignment does not work
    lines = tuple(['asset_' + str(i) for i in range(0, params[0][1])])

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    def next(self):
        for asset in range(0, self.params.n_assets):
            self.lines[asset][0] = self._owner.weights[asset]

class GetDate(bt.observer.Observer):
    lines = ('year', 'month', 'day',)

    plotinfo = dict(plot=False, subplot=False)

    def next(self):
        self.lines.year[0] = self._owner.datas[0].datetime.date(0).year
        self.lines.month[0] = self._owner.datas[0].datetime.date(0).month
        self.lines.day[0] = self._owner.datas[0].datetime.date(0).day

def calculate_portfolio_var(w, cov):
    # function that calculates portfolio risk
    return (w.T @ cov @ w)

def risk_contribution(w, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    # Marginal Risk Contribution
    MRC = cov @ w.T
    # Risk Contribution
    RC = np.multiply(MRC, w.T) / calculate_portfolio_var(w, cov)
    return RC

def target_risk_contribution(target_risk, cov):
    """
    Returns the weights of the portfolio such that the contributions to portfolio risk are as close as possiblem
    to the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    # construct the constants
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda w: np.sum(w) - 1
                        }

    def msd_risk(w, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions between weights and target_risk
        """
        w_contribs = risk_contribution(w, cov)
        return ((w_contribs - target_risk) ** 2).sum()

    w = minimize(msd_risk, init_guess,
                 args=(target_risk, cov), method='SLSQP',
                 options={'disp': False},
                 constraints=weights_sum_to_1,
                 bounds=bounds)
    return w.x

def covariances(shares, start, end):
    '''
    function that provides the covariance matrix of a certain number of shares

    :param shares: (list) shares that we would like to compute

    :return: covariance matrix
    '''
    # prices = pd.DataFrame([web.DataReader(t,'yahoo',start,end).loc[:, 'Adj Close'] for t in shares], index=shares).T.asfreq('B').ffill()
    prices = pd.DataFrame([yf.download(t,start=start,end=end).loc[:, 'Adj Close'] for t in shares], index=shares).T.asfreq('B').ffill()

    covariances = 52.0 * \
                  prices.asfreq('W-FRI').pct_change().iloc[1:, :].cov().values

    return covariances


"""
Brownian bridge
"""
def brownian_bridge(N, a, b):
    dt = 1.0 / (N-1)
    B = np.empty((1, N), dtype=np.float32)
    B[:, 0] = a-b
    for n in range(N - 2):
         t = n * dt
         xi = np.random.randn() * np.sqrt(dt)
         B[:, n + 1] = B[:, n] * (1-dt / (1-t)) + xi
    B[:, -1] = 0
    return B+b
"""
M = 1
N = 100 #Steps
B = sample_path_batch(M, N, 0, 0)
"""

def timestamp2str(ts):
    """ Converts Timestamp object to str containing date and time
    """
    date = ts.date().strftime("%Y-%m-%d")
    return date

def get_now():
    """ Return current datetime as str
    """
    return timestamp2str(datetime.datetime.now())
    # return timestamp2str(datetime.now())

def dir_exists(foldername):
    """ Return True if folder exists, else False
    """
    return os.path.isdir(foldername)

"""
load indicators for the rotational strategy and for the other strategies shown in Market signals
"""

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def load_economic_curves(start, end):
    list_fundamental = ['T10YIE', 'DFII20', 'T10Y2Y']
    df_fundamental = web.DataReader(list_fundamental, "fred", start=start, end=end)
    df_fundamental = df_fundamental.dropna()
    df_fundamental['T10YIE_T10Y2Y'] = df_fundamental['T10YIE'] - df_fundamental['T10Y2Y']
    df_fundamental = df_fundamental.drop(['T10YIE'], axis=1)

    # df_fundamental = df_fundamental.ewm(span=60,min_periods=60,adjust=False).mean()
    # df_fundamental = df_fundamental.dropna()

    df_fundamental['Max'] = df_fundamental.idxmax(axis=1)
    df_fundamental.index.name = 'Date'
    return df_fundamental

def load_fred_curve(start, end, tickers):
    df_fundamental = web.DataReader(tickers, "fred", start=start, end=end)
    df_fundamental = df_fundamental.fillna(method="ffill")
    df_fundamental.index.name = 'Date'
    return df_fundamental

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def load_riskparity_data(shares_list, startdate, enddate):
    df = pd.DataFrame()
    for i in range(len(shares_list)):
        # this_df = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]
        this_df = yf.download(shares_list[i], start=startdate, end=enddate)["Adj Close"]
        this_df = this_df.to_frame("close")
        this_df = this_df[~this_df.index.duplicated(keep='first')]
        this_df['asset'] = shares_list[i]
        df = df.append(this_df)

    # Keep common dates only
    df['Date'] = df.index
    # df.reset_index().set_index(['Date','asset'], inplace=True)
    df_pivot = df.pivot(index='Date', columns='asset', values='close')
    df_pivot = df_pivot.dropna()
    df_pivot['Date'] = df_pivot.index
    df = pd.melt(df_pivot, id_vars=['Date'])
    df = df.set_index('Date')
    df = df.rename(columns={'value': 'close'})

    return df

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def load_GEM_curves(shares_list, startdate, enddate):
    df = pd.DataFrame()
    for i in range(len(shares_list)):
        # this_df = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]
        this_df = yf.download(shares_list[i], start=startdate, end=enddate)["Adj Close"]
        this_df = this_df.to_frame("close")
        this_df = this_df[~this_df.index.duplicated(keep='first')]
        this_df['asset'] = shares_list[i]
        df = df.append(this_df)

    # Keep common dates only
    df['Date'] = df.index
    df_pivot = df.pivot(index='Date', columns='asset', values='close')
    df_pivot = df_pivot.dropna()
    df_pivot['Date'] = df_pivot.index
    df = pd.melt(df_pivot, id_vars=['Date'])
    df = df.set_index('Date')
    df = df.rename(columns={'value': 'close'})

    df['return'] = df.groupby(['asset']).pct_change(periods=params['DAYS_IN_YEAR'])
    df = df.dropna()
    df = df.drop(["close"],axis=1)
    df['date'] = df.index
    return df

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def load_AccDualMom_curves(shares_list, startdate, enddate):
    df = pd.DataFrame()
    for i in range(len(shares_list)):
        # this_df = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]
        this_df = yf.download(shares_list[i], start=startdate, end=enddate)["Adj Close"]
        this_df = this_df.fillna(method="ffill")
        this_df = this_df.to_frame("close")
        this_df = this_df[~this_df.index.duplicated(keep='first')]
        this_df['asset'] = shares_list[i]
        df = df.append(this_df)
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    # Keep common dates only
    df_pivot = df.pivot(index='Date', columns='asset', values='close')
    df_pivot = df_pivot.dropna()
    df_pivot['Date'] = df_pivot.index
    df = pd.melt(df_pivot, id_vars=['Date'])
    df = df.set_index('Date')
    df = df.rename(columns={'value': 'close'})

    last_date = df_pivot['Date'][-1]

    monthly = df.groupby(['asset']).resample('BM').last()
    monthly = monthly.droplevel('asset')
    monthly['m1'] = monthly.groupby(['asset']).close.shift(1)
    monthly['m3'] = monthly.groupby(['asset']).close.shift(3)
    monthly['m6'] = monthly.groupby(['asset']).close.shift(6)
    monthly = monthly.drop(columns=['close'])
    monthly = monthly.dropna()
    last_monthly_date = monthly.index[-1]
    monthly.rename(index={last_monthly_date: last_date}, inplace=True)

    t = pd.merge(df, monthly, how="left",  left_on=['Date','asset'], right_on = ['Date','asset'])
    t = t.bfill(axis='rows')
    t['ret1m'] = t['close']/t['m1']-1
    t['ret3m'] = t['close']/t['m3']-1
    t['ret6m'] = t['close']/t['m6']-1
    t['score'] = t['ret1m'] + t['ret3m'] + t['ret6m']
    t = t.drop(["close",'m1','m3','m6','ret1m','ret3m','ret6m'],axis=1)
    t['date'] = t.index
    return t

"""
TODO common weights functions to be used for backtesting, signals and live trading
"""

#TODO to load the data for the signals for ADM
def load_ADM_data(shares_list, startdate, enddate):
    df = pd.DataFrame()
    for i in range(len(shares_list)):
        # this_df = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]
        this_df = yf.download(shares_list[i], start=startdate, end=enddate)["Adj Close"]
        this_df = this_df.fillna(method="ffill")
        this_df = this_df.to_frame("close")
        this_df = this_df[~this_df.index.duplicated(keep='first')]
        this_df['asset'] = shares_list[i]
        df = df.append(this_df)
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    # Keep common dates only
    df_pivot = df.pivot(index='Date', columns='asset', values='close')
    df_pivot = df_pivot.dropna()
    df_pivot['Date'] = df_pivot.index
    return df_pivot

#TODO common function to calculate ADM score
def ADM_score(data, startdate, enddate):
    df_pivot = data

    df = pd.melt(df_pivot, id_vars=['Date'])
    df = df.set_index('Date')
    df = df.rename(columns={'value': 'close'})

    last_date = df_pivot['Date'][-1]

    monthly = df.groupby(['asset']).resample('BM').last()
    monthly = monthly.droplevel('asset')
    monthly['m1'] = monthly.groupby(['asset']).close.shift(1)
    monthly['m3'] = monthly.groupby(['asset']).close.shift(3)
    monthly['m6'] = monthly.groupby(['asset']).close.shift(6)
    monthly = monthly.drop(columns=['close'])
    monthly = monthly.dropna()
    last_monthly_date = monthly.index[-1]
    monthly.rename(index={last_monthly_date: last_date}, inplace=True)

    t = pd.merge(df, monthly, how="left",  left_on=['Date','asset'], right_on = ['Date','asset'])
    t = t.bfill(axis='rows')
    t['ret1m'] = t['close']/t['m1']-1
    t['ret3m'] = t['close']/t['m3']-1
    t['ret6m'] = t['close']/t['m6']-1
    t['score'] = t['ret1m'] + t['ret3m'] + t['ret6m']
    t = t.drop(["close",'m1','m3','m6','ret1m','ret3m','ret6m'],axis=1)
    t['date'] = t.index
    return t

#TODO common func for the ADM gradient_diversified_weights
def ADM_gradient_diversified_weights(score_data, reference_date):
    indicator = ['DTB3']
    momentum_df = score_data
    momentum_df['asset_weight'] = 0

    if momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > \
            momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0]:

        if momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] > \
                momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.2:
            momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 1
        elif (momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] >
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.1 and
              momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] <
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.2):
            momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 0.75
            # All weather w/o stocks
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7 * 0.25
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7 * 0.25
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7 * 0.25
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7 * 0.25

        elif (momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] <
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.1 and
              momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] >
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.1):
            momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 0.5
            # All weather w/o stocks
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7 * 0.5
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7 * 0.5
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7 * 0.5
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7 * 0.5

        elif (momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] <
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.1 and
              momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] >
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.2):
            momentum_df.loc[momentum_df['shares'] == 'VFINX', 'asset_weight'] = 0.25
            # All weather w/o stocks
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7 * 0.75
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7 * 0.75
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7 * 0.75
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7 * 0.75

        elif momentum_df.loc[momentum_df['shares'] == 'VFINX', 'score'].values[0] < \
                momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.2:
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7
    else:
        if momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] > \
                momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.2:
            momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 1
        elif (momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] >
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.1 and
              momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] <
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.2):
            momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 0.75
            # All weather w/o stocks
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7 * 0.25
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7 * 0.25
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7 * 0.25
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7 * 0.25

        elif (momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] <
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] + 0.1 and
              momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] >
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.1):
            momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 0.5
            # All weather w/o stocks
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7 * 0.5
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7 * 0.5
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7 * 0.5
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7 * 0.5

        elif (momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] <
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.1 and
              momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] >
              momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.2):
            momentum_df.loc[momentum_df['shares'] == 'VINEX', 'asset_weight'] = 0.25
            # All weather w/o stocks
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7 * 0.75
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7 * 0.75
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7 * 0.75
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7 * 0.75

        elif momentum_df.loc[momentum_df['shares'] == 'VINEX', 'score'].values[0] < \
                momentum_df.loc[pd.to_datetime(momentum_df.index).date == reference_date, indicator].values[0][0] - 0.2:
            momentum_df.loc[momentum_df['shares'] == 'TLT', 'asset_weight'] = 0.4 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'IEF', 'asset_weight'] = 0.15 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'GLD', 'asset_weight'] = 0.075 * 10 / 7
            momentum_df.loc[momentum_df['shares'] == 'GSG', 'asset_weight'] = 0.075 * 10 / 7

    return momentum_df

"""
TODO Supporting fuctions for fabio's alpha strat 
"""
def calc_idiosyncratic_mom(tseries, tspy, day, period, trading_days):
    """
    function that computes the beta: cov/var of the asset with
    respect to the SP500 index (given by SPY), which is a proxy
    for the market & the idiosyncratic momentum

    :params:
    - asset (str): the asset to consider

    :returns:
    - beta (float): the equivalent beta for the last 3 months
    """
    # get the price for the particular date
    final_day = get_trading_day(day, trading_days)

    # get the initial date
    init_day = date_minus_period(final_day, trading_days, period=period)

    # filter the dataframe for the period wanted
    filtered_prices = tseries[(pd.to_datetime(tseries.index).date >= init_day) & (pd.to_datetime(tseries.index).date <= final_day)]

    # calculate returns
    filtered_returns = filtered_prices.pct_change()
    filtered_returns = filtered_returns.fillna(0)

    # filter the dataframe for the period wanted for the SPY
    target_prices = tspy[(pd.to_datetime(tspy.index).date >= init_day) & (pd.to_datetime(tspy.index).date <= final_day)]

    # calculate the returns for the period wanted for the spy
    target_returns = target_prices.pct_change()
    target_returns = target_returns.fillna(0)

    # calculate the beta and alpha
    (beta, alpha) = stats.linregress(target_returns.values, filtered_returns.values)[0:2]

    return alpha, beta

def get_idiosyncratic_mom(assets, data, day, trading_days):
    """ function that obtains the idiosyncratic momentum for:

    - 1-month: m1
    - 3-month: m3
    - 6-month: m6
    - 12-month: m12

    of the asset list assets.

    :params:
    - assets (list): list of strings (symbols)
    - data (dataframe): dictionary of dataframe prices !! It needs to include SPY
    - day (str): date to consider

    :returns:
    - ret (dict): dictionary of returns of assets
    """
    # initialize the dictionaries
    mom = {}

    for asset in assets:
        # initialize the asset inside the return dictionary
        mom[asset] = {}

        for per in [1,3,6,12]:
            # accumulate the data
            mom[asset]['m%i' %per], _ = calc_idiosyncratic_mom(data[asset],
                                                            data['SPY'],
                                                            day,
                                                            per,
                                                            trading_days)

    return mom

def get_volatility(assets, dic_assets, day, trading_days):
    """ function that obtains the volatiliy for:
    - 1-month: m1
    - 3-month: m3
    - 6-month: m6
    - 12-months: m12

    of the asset list assets.

    :params:
    - assets (list): list of strings (symbols)
    - day (str): date to consider

    :returns:
    - ann_vol (dict): dictionary of volatility of assets
    """
    # initialize the dictionaries
    ann_vol = {}

    for asset in assets:
        # initialize the asset inside the return dictionary
        ann_vol[asset] = {}

        # for per in [1,3,6,12]:
        for per in [3,12]:
            # accumulate the data
            ann_vol[asset]['m%i' %per] = calc_volatility(dic_assets[asset],
                                                        day,
                                                        trading_days,
                                                        period=per)

    return ann_vol

def calc_volatility(tseries, today, trading_days, period=1):
    """
    Function that computes the volatility of 'tseries' from 'today'
    with window of 'period'

    :params:
    - tseries (dataframe Series): the time series to consider for the calculation
    - today (str/datetime): the date from which we should look back in time
    - period (int): period to calculate the returns on the time series

    :returns:
    - ann_vol (float): annualized volatility for tseries on period
    """

    # get the price for the particular date
    final_day = get_trading_day(today, trading_days)

    # get the initial date
    init_day = date_minus_period(final_day, trading_days, period=period)

    # filter the dataframe for the period wanted
    # filtered_prices = tseries[(tseries['Date'] >= init_day) & (tseries['Date'] <= final_day)]['Adj_Close']
    filtered_prices = tseries[(pd.to_datetime(tseries.index).date >= init_day) & (pd.to_datetime(tseries.index).date <= final_day)]

    # calculate returns
    filtered_returns = filtered_prices.pct_change()

    # calculate variance
    filtered_covariance = np.cov(filtered_returns.fillna(0).T)

    # calculate annualized volatility
    ann_vol = np.sqrt(filtered_covariance) * 100 * np.sqrt(252)

    return ann_vol

def date_minus_period(st_date, trading_days, period=1):
    """
    Function that gets the date when taking the number of
    months corresponding to the period

    :params:
    - st_date (str): initial date to remove period from
    - trading_days (list): list of trading days
    - period (int): period to calculate the returns on the time series

    :returns:
    - init_day (str): returns the initial day once the (adjusted) period
    has been taken out ('adjusted' since only future trading days are available)
    """
    # fin_day = datetime.datetime.strptime(st_date, '%Y-%m-%d')
    fin_day = st_date
    initial_day = fin_day - relativedelta(months=+period)
    # init_day = datetime.datetime.strftime(initial_day, '%Y-%m-%d')
    init_day = initial_day
    init_day = get_trading_day(init_day, trading_days)
    return init_day

def get_trading_day(st_date, trading_days, past=False):
    """
    Function that provides the first trading day
    after the date if the date is not a trading day.
    if past is True, it provides the first trading day
    before the data if the date is not a trading day

    :params:
    - st_date (str): date to test
    - trading_days (list): list of trading days
    - past (bool)

    :returns:
    - rl_date (str): might return st_date or
    couple of days later after the trading date

    """

    if is_trading_day(st_date, trading_days):
        return st_date

    else:
        rl_date = st_date
        while not is_trading_day(rl_date, trading_days):
            # dt = datetime.datetime.strptime(rl_date, '%Y-%m-%d')
            dt = rl_date
            if past:
                dt = dt - datetime.timedelta(days=1)
            else:
                dt = dt + datetime.timedelta(days=1)
            # rl_date = datetime.datetime.strftime(dt, '%Y-%m-%d')
            rl_date = dt

        return rl_date

def is_trading_day(st_date, trading_days):
    """
    Simple function that calculates if st_date is a trading day

    :params:
    - st_date (str): date to test if a trading day

    :returns:
    - boolean (bool): True if st_date is a trading day; False if not
    """

    # return st_date in trading_days
    return pd.Timestamp(st_date) in pd.to_datetime(trading_days['days']).tolist()

# for IB live trading
def AccDualMom_weights():
    shares_list = ['VFINX','VINEX','VUSTX']
    # shares_list = ['SPY','SCZ','TLT']
    AccDualMom_curves = load_AccDualMom_curves(shares_list, date.today()-timedelta(365*2.5), date.today())
    # add the 3m t bill
    t_bill = load_fred_curve(date.today()-timedelta(365*2.5), date.today(), ['DTB3'])
    t_bill = t_bill.rename(columns={"DTB3": "score"})
    t_bill['score'] = t_bill['score']/100
    t_bill['asset'] = '3m_tbill'
    #t_bill['date'] = t_bill.index
    t_bill = pd.merge(t_bill, AccDualMom_curves['date'], how='right', left_index=True, right_index=True)
    t_bill = t_bill.fillna(method="ffill")
    t_bill = t_bill.fillna(method="bfill")
    AccDualMom_curves = AccDualMom_curves.append(t_bill)

    end_month = AccDualMom_curves.groupby(AccDualMom_curves.index.month).max().sort_values('date').iloc[-2]['date']
    AccDualMom_stocks = [['equity',shares_list[0], 'Vanguard 500 Index Fund Investor Shares', AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[0]) & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[0]) & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/VFINX'],
                  ['equity_intl',shares_list[1], 'Vanguard International Explorer Fund Investor Shares', AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[1]) & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[1]) & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0],  'https://finance.yahoo.com/quote/VINEX'],
                  ['3m_bills','DTB3', '3-Month Treasury Bill Secondary Market Rate', AccDualMom_curves.loc[(AccDualMom_curves.asset == '3m_tbill') & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == '3m_tbill') & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0], 'https://fred.stlouisfed.org/series/DTB3']]

    AccDualMom_stocks_df = pd.DataFrame(AccDualMom_stocks, columns=['type','ticker', 'name', 'month end score','today score','link'])

    score_col_name = 'today score'
    data = AccDualMom_stocks_df

    if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
            data.loc[data['type'] == 'equity_intl', score_col_name].values[0]:
        if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
                data.loc[data['type'] == '3m_bills', score_col_name].values[0]:
            ticker_eom = data.loc[data['type'] == 'equity', 'ticker'].values[0]
        else:
            ticker_eom = "VUSTX"
    else:
        if data.loc[data['type'] == 'equity_intl', score_col_name].values[0] > \
                data.loc[data['type'] == '3m_bills', score_col_name].values[0]:
            ticker_eom = data.loc[data['type'] == 'equity_intl', 'ticker'].values[0]
        else:
            ticker_eom = "VUSTX"

    if ticker_eom == 'VFINX':
        ticker_eom = 'VOO'
    elif ticker_eom == 'VINEX':
        ticker_eom = 'VSS'
    elif ticker_eom == 'VUSTX':
        ticker_eom = 'TLT'

    w={'contractDesc':['VOO','VSS','TLT'],'conid':['136155102','59234393','15547841'],'target_allocation':[0,0,0]}
    weights = pd.DataFrame(data=w)
    weights.loc[weights['contractDesc'] == ticker_eom, 'target_allocation'] = 1

    return weights

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def load_vigilant_curves(shares_list, startdate, enddate):
    df = pd.DataFrame()
    for i in range(len(shares_list)):
        # this_df = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]
        this_df = yf.download(shares_list[i], start=startdate, end=enddate)["Adj Close"]
        this_df = this_df.to_frame("close")
        this_df = this_df[~this_df.index.duplicated(keep='first')]
        this_df['asset'] = shares_list[i]
        df = df.append(this_df)

    # Keep common dates only
    df['Date'] = df.index
    df_pivot = df.pivot(index='Date', columns='asset', values='close')
    df_pivot = df_pivot.dropna()
    df_pivot['Date'] = df_pivot.index
    df = pd.melt(df_pivot, id_vars=['Date'])
    df = df.set_index('Date')
    df = df.rename(columns={'value': 'close'})

    last_date = df_pivot['Date'][-1]

    monthly = df.groupby(['asset']).resample('BM').last()
    monthly = monthly.droplevel('asset')
    monthly['m1'] = monthly.groupby(['asset']).close.shift(1)
    monthly['m3'] = monthly.groupby(['asset']).close.shift(3)
    monthly['m6'] = monthly.groupby(['asset']).close.shift(6)
    monthly['m12'] = monthly.groupby(['asset']).close.shift(12)
    monthly = monthly.drop(columns=['close'])
    monthly = monthly.dropna()
    last_monthly_date = monthly.index[-1]
    monthly.rename(index={last_monthly_date: last_date}, inplace=True)

    t = pd.merge(df, monthly, how="left",  left_on=['Date','asset'], right_on = ['Date','asset'])
    t = t.bfill(axis='rows')
    t['ret1m'] = t['close']/t['m1']-1
    t['ret3m'] = t['close']/t['m3']-1
    t['ret6m'] = t['close']/t['m6']-1
    t['ret6m'] = t['close']/t['m6']-1
    t['ret12m'] = t['close']/t['m12']-1
    t['score'] = 12*t['ret1m'] + 4*t['ret3m'] + 2*t['ret6m'] + t['ret12m']
    t = t.drop(["close",'m1','m3','m6','ret1m','ret3m','ret6m'],axis=1)
    t['date'] = t.index

    return t
"""
Functions to calculate performance metrics

From: http://www.turingfinance.com/computational-investing-with-python-week-one/
http://quantopian.github.io/empyrical/_modules/empyrical/stats.html (for omega ratio and validation)

"""

"""
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.
"""


def vol(returns):
    # Return the standard deviation of returns
    return np.std(returns)


def beta(returns, market):
    # Create a matrix of [returns, market]
    m = np.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    if np.std(market) != 0:
        return np.cov(m)[0][1] / np.std(market)
    else:
        return math.nan


def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the difference to the power of order
    return np.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    # This method returns a higher partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def var(returns, alpha):
    # This method calculates the historical simulation var of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])


def cvar(returns, alpha):
    # This method calculates the condition VaR of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    if index > 0:
        return abs(sum_var / index)
    else:
        return math.nan


def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)


def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)

    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)

def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def average_dd(returns, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def average_dd_squared(returns, periods):
    # Returns the average maximum drawdown squared over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = math.pow(dd(returns, i), 2.0)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def treynor_ratio(er, returns, market, rf):
    if beta(returns, market) != 0:
        return (er - rf) / beta(returns, market)
    else:
        return math.nan

def sharpe_ratio(er, returns, rf):
    if vol(returns) != 0:
        return (er - rf) / vol(returns)
    else:
        return math.nan

def information_ratio(returns, benchmark):
    diff = returns - benchmark
    if vol(diff) != 0:
        return np.mean(diff) / vol(diff)
    else:
        return math.nan


def modigliani_ratio(er, returns, benchmark, rf):
    np_rf = np.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf


def excess_var(er, returns, rf, alpha):
    if var(returns, alpha) != 0:
        return (er - rf) / var(returns, alpha)
    else:
        return math.nan


def conditional_sharpe_ratio(er, returns, rf, alpha):
    if cvar(returns, alpha) != 0:
        return (er - rf) / cvar(returns, alpha)
    else:
        return math.nan


def omega_ratio(er, returns, rf, target=0):
    """
    Omega ratio definition replaced by the definition found in http://quantopian.github.io/empyrical/_modules/empyrical/stats.html
    that matches the Wikipedia definition https://en.wikipedia.org/wiki/Omega_ratio

    old definition:

    def omega_ratio(er, returns, rf, target=0):
        return (er - rf) / lpm(returns, target, 1)
    """
    if lpm(returns, target+rf, 1) != 0:
        return hpm(returns, target+rf, 1)/lpm(returns, target+rf, 1)
    else:
        return math.nan


def sortino_ratio(er, returns, rf, target=0):
    if math.sqrt(lpm(returns, target, 2)) != 0:
        return (er - rf) / math.sqrt(lpm(returns, target, 2))
    else:
        return math.nan


def kappa_three_ratio(er, returns, rf, target=0):
    if math.pow(lpm(returns, target, 3), float(1 / 3)) != 0:
        return (er - rf) / math.pow(lpm(returns, target, 3), float(1 / 3))
    else:
        return math.nan


def gain_loss_ratio(returns, target=0):
    if lpm(returns, target, 1) != 0:
        return hpm(returns, target, 1) / lpm(returns, target, 1)
    else:
        return math.nan


def upside_potential_ratio(returns, target=0):
    if math.sqrt(lpm(returns, target, 2)) != 0:
        return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))
    else:
        return math.nan


def calmar_ratio(er, returns, rf):
    if max_dd(returns)!=0:
        return (er - rf) / max_dd(returns)
    else:
        return math.nan

def sterling_ration(er, returns, rf, periods):
    if average_dd(returns, periods)!=0:
        return (er - rf) / average_dd(returns, periods)
    else:
        return math.nan


def burke_ratio(er, returns, rf, periods):
    if math.sqrt(average_dd_squared(returns, periods))!=0:
        return (er - rf) / math.sqrt(average_dd_squared(returns, periods))
    else:
        return math.nan

"""
def test_risk_metrics():
    # This is just a testing method
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    print("vol =", vol(r))
    print("beta =", beta(r, m))
    print("hpm(0.0)_1 =", hpm(r, 0.0, 1))
    print("lpm(0.0)_1 =", lpm(r, 0.0, 1))
    print("VaR(0.05) =", var(r, 0.05))
    print("CVaR(0.05) =", cvar(r, 0.05))
    print("Drawdown(5) =", dd(r, 5))
    print("Max Drawdown =", max_dd(r))


def test_risk_adjusted_metrics():
    # Returns from the portfolio (r) and market (m)
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    # Expected return
    e = np.mean(r)
    # Risk free rate
    f = 0.06
    # Risk-adjusted return based on Volatility
    print("Treynor Ratio =", treynor_ratio(e, r, m, f))
    print("Sharpe Ratio =", sharpe_ratio(e, r, f))
    print("Information Ratio =", information_ratio(r, m))
    # Risk-adjusted return based on Value at Risk
    print("Excess VaR =", excess_var(e, r, f, 0.05))
    print("Conditional Sharpe Ratio =", conditional_sharpe_ratio(e, r, f, 0.05))
    # Risk-adjusted return based on Lower Partial Moments
    print("Omega Ratio =", omega_ratio(e, r, f))
    print("Sortino Ratio =", sortino_ratio(e, r, f))
    print("Kappa 3 Ratio =", kappa_three_ratio(e, r, f))
    print("Gain Loss Ratio =", gain_loss_ratio(r))
    print("Upside Potential Ratio =", upside_potential_ratio(r))
    # Risk-adjusted return based on Drawdown risk
    print("Calmar Ratio =", calmar_ratio(e, r, f))
    print("Sterling Ratio =", sterling_ration(e, r, f, 5))
    print("Burke Ratio =", burke_ratio(e, r, f, 5))


if __name__ == "__main__":
    test_risk_metrics()
    test_risk_adjusted_metrics()
"""

"""
utils for IBKR
"""

FX_ticker_mapping = pd.DataFrame(list(zip(["EUR",    "USD",    "CHF",   "USD",     "EUR",    "CHF"],
                                          ["USD",    "EUR",    "USD",   "CHF",     "CHF",    "EUR"],
                                          ["EUR.USD","EUR.USD","CHF.USD","CHF.USD","EUR.CHF","EUR.CHF"],
                                          ["SELL",   "BUY",    "SELL",   "BUY",    "SELL",   "BUY"])),
                                      columns=['ccyfrom', 'ccyto', 'ticker', 'side'])


def send_telegram(text):
    token = '1679672381:AAEq2sml5nj17YSHAgKsPq5t560nUnc0Jz0'
    params = {'chat_id': 1132052090, 'text': text, 'parse_mode': 'HTML'}
    resp = requests.post('https://api.telegram.org/bot{}/sendMessage'.format(token), params)
    resp.raise_for_status()

def log(txt):
    ''' Logging function for this strategy txt is the statement and dt can be used to specify a specific datetime'''
    st.write("%s, %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), txt))

def mycOID(ticker='EUR.USD', side = 'BUY'):
    return ticker + "_" + side + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")

def ib_exit(ib_client):
    ib_client.logout()
    ib_client.close_session()

# Orders functions
#this_order_status = order_status[0][0]['order_status']

def order_status_management(ib_client, order_status, order_details):
    amt = order_details[2]['orders'][0]['quantity']
    side = order_details[2]['orders'][0]['side']
    ticker = order_details[2]['orders'][0]['ticker']
    type = order_details[2]['orders'][0]['orderType']

    if order_status == 'Filled':
        # Print and message in the console and send a message to telegram
        msg = side + " " + type + " order executed for " + str(amt) + " shares of " + ticker +". Program continues to run."
        log(msg)
        send_telegram(msg)
        return True
    elif order_status == 'ApiPending':
        # Wait 5 minutes and then recheck the state. If state is still ApiPending, then exit.
        if order_status_management.count == 5:
            msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + " is ApiPending. Program exits."
            log(msg)
            ib_exit(ib_client)
        sleep(30 - time() % 30)
        ib_client.tickle()
        order_status_management.count += 1
        live_orders = ib_client.get_live_orders()
        live_orders_df = pd.DataFrame.from_records(live_orders['orders'])
        current_status = \
        live_orders_df.loc[live_orders_df['orderId'] == int(order_details[0][0]['order_id']), 'status'].values[0]
        order_status_management(ib_client, current_status, order_details)
    elif order_status == 'ApiCancelled':
        msg = side + " " + type + " order for " + str(amt) + " shares of " + ticker + " was cancelled as requested. " \
              "Status is ApiCancelled. Program continues."
        log(msg)
        send_telegram(msg)
        return True
    elif order_status == 'Inactive':
        msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + \
              " is Inactive. Program exits."
        log(msg)
        send_telegram(msg)
        ib_exit(ib_client)
    elif order_status == 'PendingSubmit':
        msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + \
              " is PendingSubmit. Likely this is because the exchange is closed. Program continues."
        log(msg)
        send_telegram(msg)
        return True
    elif order_status == 'PendingCancel':
        msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + " is " + order_status + ". Program continues."
        log(msg)
        send_telegram(msg)
        return True
    elif order_status == 'PreSubmitted':
        msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + \
              " is Presubmitted. Program continues."
        log(msg)
        send_telegram(msg)
        return True
    elif order_status == 'Submitted':
        msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + \
              " is Submitted. Program continues."
        log(msg)
        send_telegram(msg)
        return True
    elif order_status == 'Cancelled':
        msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + \
              " is Cancelled. This is likely due to not sufficient balance. Program exits."
        log(msg)
        send_telegram(msg)
        ib_exit(ib_client)
    else:
        # Wait 5 minutes and then recheck the state. If state is still ApiPending, then exit.
        if order_status_management.count == 5:
            msg = "Status of " + side + " " + type + " order for " + str(amt) + " shares of " + ticker + " is " + order_status + ". Program exits."
            log(msg)
            send_telegram(msg)
            ib_exit(ib_client)

        sleep(30 - time() % 30)
        ib_client.tickle()
        order_status_management.count += 1
        live_orders = ib_client.get_live_orders()
        live_orders_df = pd.DataFrame.from_records(live_orders['orders'])
        current_status = \
        live_orders_df.loc[live_orders_df['orderId'] == int(order_details[0][0]['order_id']), 'status'].values[0]
        order_status_management(ib_client, current_status, order_details)

#order_status_management.count = 0

def get_FXconid(ib_client, ticker='EUR.USD'):
    if not ticker in list(FX_ticker_mapping['ticker']):
        log("Error. Ticker is not valid. Please consider the ticker obtained reversing the currency order.")
        ib_exit(ib_client)
    sym_content = ib_client.symbol_search(symbol=ticker)
    sym_content = sym_content[0]['sections'][:]
    sym_df = pd.DataFrame.from_records(sym_content)
    conid = sym_df.loc[sym_df['secType']=='CASH']['conid'].values[0]
    return int(conid)

def get_FX_quote(ib_client, ticker='EUR.USD'):
    if not ticker in list(FX_ticker_mapping['ticker']):
        log("Error. Ticker is not valid. Please consider the ticker obtained reversing the currency order.")
        ib_exit(ib_client)
    conid = get_FXconid(ib_client, ticker)
    quote = ib_client.market_data(conids=conid, since='0', fields=['31', '84', '86'])
    quote = ib_client.market_data(conids=conid, since='0', fields=['31', '84', '86'])
    if '31' not in quote[0]:
        log("Error: " + str(ticker) + " quote not available. Program exiting.")
        ib_exit(ib_client)
    else:
        quote = float(quote[0]['31'].replace(',', ''))
    return quote

def get_stkConid(ib_client, ticker='QQQ'):
     sym_content = ib_client.symbol_search(symbol=ticker)
     sym_content = sym_content[0]['sections'][:]
     sym_df = pd.DataFrame.from_records(sym_content)
     conid = sym_df.loc[sym_df['secType']=='STK']['conid'].values
     return int(conid[0])

def myround(x, prec=2, base=.05):
  return round(base * round(float(x)/base),prec)

def get_minIncrement(ib_client, conid=''):
    info = ib_client.contract_details_rules(conid=conid)
    return float(info['rules']['increment'])

def get_rth(ib_client, conid=''):
    info = ib_client.contract_details_rules(conid=conid)
    return info['r_t_h']

def get_stk_quote(ib_client, conid = ''): # stock quote as mid price
    rth_flag = get_rth(ib_client, conid)
    if not rth_flag:
        log("Contract " + str(conid) + " is outside regular market hours.")
    try:
        stk_quote = ib_client.market_data(conid, since='0', fields=['31', '84', '86'])
        stk_quote = ib_client.market_data(conid, since='0', fields=['31', '84', '86'])
        bid = float(stk_quote[0]['84'].replace(',', ''))  # IB uses commas as thousands separators
        ask = float(stk_quote[0]['86'].replace(',', ''))
        mid = (bid + ask) / 2
        return mid
    except:
        if not rth_flag:
            st.write("No quote could be downloaded for contract " + str(conid) + ", because it is outside regular "
                                                                                    "market hours.")
            return 0
        else:
            st.write("No quote could be downloaded for contract " + str(conid) + ", with reason unknown.")
            return 0

def convertFX(ib_client, ccyfrom='EUR', ccyto='USD', amt_ccyfrom=0, account_id=''):
    condition = (FX_ticker_mapping['ccyfrom'] == ccyfrom) & (FX_ticker_mapping['ccyto'] == ccyto)
    ib_ticker = FX_ticker_mapping.loc[condition,'ticker'].values[0]
    side = FX_ticker_mapping.loc[condition,'side'].values[0]

    conid = get_FXconid(ib_client, ticker=ib_ticker)
    mycOID_ = mycOID(ticker=ib_ticker, side=side)
    orders = {'orders':
            [{
              'acctId': account_id,
              'conid': conid,
              'secType': 'CASH',
              'cOID': mycOID_,
              'parentId': mycOID_,
              'orderType': 'MKT',
              'listingExchange': 'IDEALPRO',
              #'outsideRTH': True,
              #'price': 0,
              'side': side,
              'ticker': ib_ticker,
              'tif': 'GTC',
              #'referrer': 'string',
              'quantity': round(amt_ccyfrom-1),
              'useAdaptive': True
            }]}
    order_status = ib_client.place_orders(account_id, orders)
    message_flag = False

    if 'error' in order_status:
        return [order_status, mycOID_, orders]

    while 'message' in order_status[0]:
        message_flag = True
        reply=st.radio(''.join(map(str, order_status[0]['message'])) , ('Yes', 'No'))
        if reply == "Yes":
            reply_ib = True
        else:
            reply_ib = False
        reply_id=order_status[0]['id']
        order_status=ib_client.place_order_reply(reply_id=reply_id, reply=reply_ib)
    if not 'order_status' in order_status[0]:
        if message_flag == False:
            log('Error: no order_status delivered and no message provided.')
        else:
            log('Error: no order_status delivered after messages were replied.')
        ib_exit(ib_client)
    #log("Order " + order_status[0]['order_status'] + ", order_id: " +  order_status[0]['order_id'] + ", " + order['side'] + " " + str(order['quantity']) + " " + order['ticker'])
    return [order_status, mycOID_, orders]

def stock_buysell(ib_client, ib_ticker='QQQ', conid='', amt=0, account_id='', side = 'BUY', price=0):
    mycOID_ = mycOID(ticker=ib_ticker, side = side)
    orders = {'orders': [{
              'acctId': account_id,
              'conid': int(conid),
              'secType': 'STK',
              'cOID': mycOID_,
              'parentId': mycOID_,
              'orderType': 'LMT',
              'listingExchange': 'SMART',
              'outsideRTH': True,
              'price': price,
              'side': side,
              'ticker': ib_ticker,
              'tif': 'GTC',
              #'referrer': 'string',
              'quantity': round(amt),
              'useAdaptive': True
            }]}
    order_status = ib_client.place_orders(account_id, orders)
    message_flag = False

    if 'error' in order_status:
        return [order_status, mycOID_, orders]

    while 'message' in order_status[0]:
        message_flag = True
        reply=st.radio(''.join(map(str, order_status[0]['message'])), ('Yes', 'No'), key= mycOID(ticker=ib_ticker, side = side))
        if reply == 'Yes':
            reply_ib = True
        else:
            reply_ib = False
        reply_id=order_status[0]['id']
        order_status=ib_client.place_order_reply(reply_id=reply_id, reply=reply_ib)
    if not 'order_status' in order_status[0]:
        if message_flag == False:
            log('Error: no order_status delivered and no message provided.')
        else:
            log('Error: no order_status delivered after messages were replied.')
        ib_exit(ib_client)
    #log("Order " + order_status[0]['order_status'] + ", order_id: " +  order_status[0]['order_id'] + ", " + order['side'] + " " + str(order['quantity']) + " " + order['ticker'])
    return [order_status, mycOID_, orders]

def get_portfolio_returns(ib_client, account_id, freq = 'D'):
    perf_data = ib_client.portfolio_performance(account_id, freq)
    returns = pd.DataFrame(data={'date': perf_data['cps']['dates'], 'returns':perf_data['cps']['data'][0]['returns']})
    returns['returns'] = (returns['returns'] + 1).pct_change()
    returns['returns'].iloc[0] = 0
    returns['date'] = pd.to_datetime(returns['date'], format="%Y-%m-%d")
    returns['date'] = returns['date'].dt.date
    return returns

def get_portfolio_nav(ib_client, account_id, freq = 'D'):
    perf_data = ib_client.portfolio_performance(account_id, freq)
    nav = pd.DataFrame(data={'date': perf_data['nav']['dates'], 'nav':perf_data['nav']['data'][0]['navs']})
    nav['date'] = pd.to_datetime(nav['date'], format="%Y-%m-%d")
    nav['date'] = nav['date'].dt.date
    return nav

def live_portfolio_metrics(params, returns_df):
    cumret = (returns_df + 1).cumprod()
    returns = returns_df.to_numpy()
    tot_return = (returns + 1).prod() - 1
    annual_return = ((1 + tot_return) ** (params['DAYS_IN_YEAR'] / (round((params['enddate'] - params['startdate']).days * params['DAYS_IN_YEAR'] / 365))) - 1)
    max_dd = -np.nanmin((cumret.shift(-1) / cumret.cummax(axis=0) - 1).shift(1))
    max_money_dd = -np.nanmin((cumret.shift(-1) - cumret.cummax(axis=0)).shift(1))

    target = params['targetrate']
    rate = params['riskfree']
    alpha = params['alpha']
    market_mu = params['market_mu']
    market_sigma = params['market_sigma']

    factor = params['DAYS_IN_YEAR']

    rate = pow(1.0 + rate, 1.0 / factor) - 1.0
    market_mu = pow(1.0 + market_mu, 1.0 / factor) - 1.0
    market_sigma = market_sigma / np.sqrt(factor)


    # Simulate market returns following a geometric brownian motion with used specified mu and sigma
    dt = 1
    market_returns = market_mu * dt + market_sigma * np.sqrt(dt) * np.random.normal(0, 1, len(returns))

    ret_avg = returns_df.mean()
    ret_dev = returns_df.std()
    ret_skew = skew(returns)
    ret_kurt = kurtosis(returns)

    treynor_ratio_ = treynor_ratio(ret_avg, returns, market_returns, rate)
    sharpe_ratio_ = sharpe_ratio(ret_avg, returns, rate)
    information_ratio_ = information_ratio(returns, market_returns)
    var_ = var(returns, alpha)
    cvar_ = cvar(returns, alpha)
    excess_var_ = excess_var(ret_avg, returns, rate, alpha)
    conditional_sharpe_ratio_ = conditional_sharpe_ratio(ret_avg, returns, rate, alpha)
    omega_ratio_ = omega_ratio(ret_avg, returns, rate, target)
    sortino_ratio_ = sortino_ratio(ret_avg, returns, rate, target)
    kappa_three_ratio_ = kappa_three_ratio(ret_avg, returns, rate, target)
    gain_loss_ratio_ = gain_loss_ratio(returns, target)
    upside_potential_ratio_ = upside_potential_ratio(returns, target)
    calmar_ratio_ = calmar_ratio(ret_avg, returns, rate)

    if params['annualize'] and factor is not None:
        ret_avg = ret_avg * factor
        ret_dev = ret_dev * np.sqrt(factor)
        ret_skew = ret_skew / np.sqrt(factor)
        ret_kurt = ret_kurt / factor
        # A factor was found -> annualize the quantities
        treynor_ratio_ = treynor_ratio_ * np.sqrt(factor)
        sharpe_ratio_ = sharpe_ratio_ * np.sqrt(factor)
        information_ratio_ = information_ratio_ * np.sqrt(factor)
        var_ = var_ * np.sqrt(factor)
        cvar_ = cvar_ * np.sqrt(factor)
        excess_var_ = excess_var_ * np.sqrt(factor)
        conditional_sharpe_ratio_ = conditional_sharpe_ratio_ * np.sqrt(factor)
        omega_ratio_ = omega_ratio_
        sortino_ratio_ = sortino_ratio_ * np.sqrt(factor)
        kappa_three_ratio_ = kappa_three_ratio_ * np.sqrt(factor)
        gain_loss_ratio_ = gain_loss_ratio_
        upside_potential_ratio_ = upside_potential_ratio_
        calmar_ratio_ = calmar_ratio_ * factor

    kpis = {  # PnL
        'Starting cash': params['initial_cash'],
        'End value': params['initial_cash'] * (1 + tot_return),
        'Total return': tot_return,
        'Annual return': annual_return,
        'Annual return (asset mode)': annual_return,
        'Max money drawdown': max_money_dd,
        'Max percentage drawdown': max_dd,
        # Distribution
        'Returns volatility': ret_dev,
        'Returns skewness': ret_skew,
        'Returns kurtosis': ret_kurt,
        # Risk-adjusted return based on Volatility
        'Treynor ratio': treynor_ratio_,
        'Sharpe ratio': sharpe_ratio_,
        'Information ratio': information_ratio_,
        # Risk-adjusted return based on Value at Risk
        'VaR': var_,
        'Expected Shortfall': cvar_,
        'Excess var': excess_var_,
        'Conditional sharpe ratio': conditional_sharpe_ratio_,
        # Risk-adjusted return based on Lower Partial Moments
        'Omega ratio': omega_ratio_,
        'Sortino ratio': sortino_ratio_,
        'Kappa three ratio': kappa_three_ratio_,
        'Gain loss ratio': gain_loss_ratio_,
        'Upside potential ratio': upside_potential_ratio_,
        # Risk-adjusted return based on Drawdown risk
        'Calmar ratio': calmar_ratio_
    }

    kpis_df = pd.DataFrame.from_dict(kpis, orient='index')

    kpis_df['Category'] = ['P&L', 'P&L', 'P&L', 'P&L', 'P&L',
                           'Risk-adjusted return based on Drawdown', 'Risk-adjusted return based on Drawdown',
                           'Distribution moments', 'Distribution moments', 'Distribution moments',
                           'Risk-adjusted return based on Volatility',
                           'Risk-adjusted return based on Volatility',
                           'Risk-adjusted return based on Volatility',
                           'Risk-adjusted return based on Value at Risk',
                           'Risk-adjusted return based on Value at Risk',
                           'Risk-adjusted return based on Value at Risk',
                           'Risk-adjusted return based on Value at Risk',
                           'Risk-adjusted return based on Lower Partial Moments',
                           'Risk-adjusted return based on Lower Partial Moments',
                           'Risk-adjusted return based on Lower Partial Moments',
                           'Risk-adjusted return based on Lower Partial Moments',
                           'Risk-adjusted return based on Lower Partial Moments',
                           'Risk-adjusted return based on Drawdown']

    kpis_df['Metrics'] = kpis_df.index

    all_stats = kpis_df.set_index(['Category', 'Metrics'])

    all_stats.columns = ['live portfolio']
    return all_stats