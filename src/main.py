from GLOBAL_VARS import *
from utils import *
from strategies import *
import datetime
import argparse
import backtrader as bt
from report import Cerebro
import report_aggregator
import sys
pd.options.mode.chained_assignment = None  # default='warn'
from PortfolioDB import PortfolioDB
import platform
import streamlit as st

# Strategy parameters not passed
from strategies import customweights

def runOneStrat(strategy=None):

    # Set the timeframe
    if params['historic'] == 'long':
        timeframe = bt.TimeFrame.Years
    else:
        timeframe = bt.TimeFrame.Days

    params['weights'] = list(filter(None, params['weights']))


    # Set the minimum periods, lookback period (and other parameters) depending on the data used (daily or yearly) and
    # if weights are used instead of a strategy
    if params['weights'] != [] or strategy == 'customweights':
        corrmethod = params['corrmethod_custweights']
        reb_period = params['reb_period_custweights']
        lookback_period_short = params['lookback_period_short_custweights']
        lookback_period_long = params['lookback_period_long_custweights']
        moving_average_period = params['moving_average_period_custweights']
        momentum_period = params['momentum_period_custweights']
        momentum_percentile = params['momentum_percentile_custweights']
    elif timeframe == bt.TimeFrame.Days:
        corrmethod = params['corrmethod_days']
        reb_period = params['reb_period_days']
        lookback_period_short = params['lookback_period_short_days']
        lookback_period_long = params['lookback_period_long_days']
        moving_average_period = params['moving_average_period_days']
        momentum_period = params['momentum_period_days']
        momentum_percentile = params['momentum_percentile_days']
    elif timeframe == bt.TimeFrame.Years:
        corrmethod = params['corrmethod_years']
        reb_period = params['reb_period_years']
        lookback_period_short = params['lookback_period_short_years']
        lookback_period_long = params['lookback_period_long_years']
        moving_average_period = params['moving_average_period_years']
        momentum_period = params['momentum_period_years']
        momentum_percentile = params['momentum_percentile_years']

    max_lookback = max(lookback_period_short, lookback_period_long, moving_average_period, momentum_period)

    # startdate = datetime.datetime.strptime(params['startdate'], "%Y-%m-%d")
    # enddate = datetime.datetime.strptime(params['enddate'], "%Y-%m-%d")
    startdate = params['startdate'] - datetime.timedelta(round(max_lookback*365/params['DAYS_IN_YEAR']))
    enddate = params['enddate']

    # Adjust the user input
    shares_list = list(filter(None, params['shares']))
    shareclass = list(filter(None, params['shareclass']))

    # Initialize the engine
    cerebro = Cerebro(cheat_on_open=True, timeframe=timeframe)
    # cerebro = Cerebro()
    # cerebro.broker.set_coo(True)
    # cerebro.broker.set_coc(True)
    cerebro.broker.set_cash(params['initial_cash'])
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.set_shortcash(True)  # Can short the cash
    # cerebro.broker.setcommission(leverage=params['leverage'], interest=0.05)

    # Add the data
    data = []
    if params['historic'] == 'medium':
        # Import the historical assets
        for share in shares_list:
            df = import_histprices_db(share)
            for column in ['open', 'high', 'low', 'close']:
                df[column] = add_expenses(df[column], expense_ratio=params['expense_ratio'], timeframe=timeframe)
            for column in ['open', 'high', 'low']:
                df[column] = df['close']
            df['volume'] = 0
            data.append(df)

        if params['shareclass'] == '':
            db = PortfolioDB(databaseName=DB_NAME)
            shareclass = []
            for share in shares_list:
                thisshareinfo = db.getStockInfo(share)
                shareclass.append(thisshareinfo['asset_class'].values[0])

    elif params['historic'] == 'long':
        # Import the historical assets
        for share in shares_list:
            df = import_histprices_db(share)
            for column in ['open', 'high', 'low', 'close']:
                df[column] = add_expenses(df[column], expense_ratio=params['expense_ratio'], timeframe=timeframe)
            for column in ['open', 'high', 'low']:
                df[column] = df['close']
            df['volume'] = 0
            data.append(df)

        if params['shareclass'] == '':
            db = PortfolioDB(databaseName=DB_NAME)
            shareclass = []
            for share in shares_list:
                thisshareinfo = db.getStockInfo(share)
                shareclass.append(thisshareinfo['asset_class'].values[0])
    else:
        # download the datas
        for i in range(len(shares_list)):
            if shares_list[i] == "CASH":
                idx = pd.date_range(start=startdate, end=enddate)
                this_assets = pd.Series(data=1, index=idx)
            else:
                # this_assets = web.DataReader(shares_list[i], "yahoo", startdate, enddate)["Adj Close"]
                this_assets = yf.download(shares_list[i], start=startdate, end=enddate)["Adj Close"]

            if APPLY_LEVERAGE_ON_LIVE_STOCKS == True:
                this_assets = add_expenses(this_assets, expense_ratio=params['expense_ratio'],timeframe=timeframe).to_frame("close")
            else:
                this_assets = this_assets.to_frame("close")

            for column in ['open', 'high', 'low']:
                this_assets[column] = this_assets['close']
            this_assets['volume'] = 0
            data.append(this_assets)

    # if params['indicator'] : # if params['rotationstrat'] or params['rot_adm'] or params['acc_dualmom'] params['acc_dualmom2']:
    if params['rotationstrat'] or params['rot_adm'] or params['acc_dualmom'] or params['acc_dualmom2'] or params['rot_adm_dual_rp'] or params['TESTING'] \
            or params['specific_vigilant'] or params['specific_rot'] or params['specific_rot2'] or params['specific_adm'] or params['specific_adm_grad_div'] or params['specific_fabio_adm2'] \
            or params['specific_fabiofg_adm2'] or params['specific_fabio_adm3']:
        # now import the non-tradable indexes for the rotational strategy
        indicatorLabels = ['DFII20', 'T10Y2Y', 'T10YIE_T10Y2Y', 'DTB3']
        rot_indicators = load_economic_curves(startdate, enddate)
        t_bill = load_fred_curve(startdate, enddate, indicatorLabels[3])
        t_bill['DTB3'] = t_bill['DTB3']/100
        rot_indicators = pd.merge(rot_indicators[['DFII20', 'T10Y2Y', 'T10YIE_T10Y2Y']], t_bill, how="left", left_index=True, right_index=True)

        for indicatorLabel in indicatorLabels:
            df = rot_indicators[[indicatorLabel]]
            for column in ['open', 'high', 'low', 'close']:
                df[column] = df[indicatorLabel]

            df['volume'] = 0
            df = df[['open', 'high', 'low', 'close', 'volume']]
            #data.append(bt.feeds.PandasData(dataname=df, fromdate=startdate, todate=enddate, timeframe=bt.TimeFrame.Days))
            data.append(df)

        shareclass = shareclass + ['non-tradable', 'non-tradable', 'non-tradable', 'non-tradable']
        shares_list = shares_list + indicatorLabels

    if params['benchmark'] != '':
        if params['historic'] == 'medium' or params['historic'] == 'long':
            # look for the benchmark in the database
            #benchmark_df = import_process_hist(params['benchmark, args) # First look for the benchmark in the historical "database"
            benchmark_df = import_histprices_db(params['benchmark'])
        else:
            # benchmark_df = web.DataReader(params['benchmark'], "yahoo", startdate, enddate)["Adj Close"]
            benchmark_df = yf.download(params['benchmark'], start=startdate, end=enddate)["Adj Close"]

            benchmark_df = benchmark_df.to_frame("close")

        if benchmark_df is not None:
            for column in ['open', 'high', 'low']:
                benchmark_df[column] = benchmark_df['close']

            benchmark_df['volume'] = 0

        #data.append(bt.feeds.PandasData(dataname=benchmark_df, fromdate=startdate, todate=enddate, timeframe=timeframe))
        data.append(benchmark_df)

        shareclass = shareclass + ['benchmark']
        shares_list = shares_list + [params['benchmark']]

    if params['leverage'] > 0:
        cash_df = get_loan(startdate, enddate, BM_rate_flg, params['interest_rate'])

        if cash_df is not None:
            for column in ['open', 'high', 'low']:
                cash_df[column] = cash_df['close']
            cash_df['volume'] = 0

        # data.append(bt.feeds.PandasData(dataname=benchmark_df, fromdate=startdate, todate=enddate, timeframe=timeframe))
        data.append(cash_df)

        shareclass = shareclass + ['loan']
        shares_list = shares_list + ['loan']

    data = common_dates(data=data, fromdate=startdate, todate=enddate, timeframe=timeframe)

    i = 0
    for dt in data:
        dt_feed = bt.feeds.PandasData(dataname=dt, fromdate=startdate, todate=enddate, timeframe=timeframe)
        cerebro.adddata(dt_feed, name=shares_list[i])
        i = i + 1

    n_assets = len([x for x in shareclass if x not in ['non-tradable', 'benchmark', 'loan']])
    cerebro.addobserver(targetweightsobserver, n_assets=n_assets)
    cerebro.addobserver(effectiveweightsobserver, n_assets=n_assets)

    # if you provide the weights, use them
    if params['weights'] != [] and strategy == 'customweights':
        weights_list = params['weights']
        weights_listt = [float(i) for i in weights_list]

        cerebro.addstrategy(customweights,
                            n_assets=n_assets,
                            shares=shares_list,
                            initial_cash=params['initial_cash'],
                            contribution=params['contribution'],
                            assetweights=weights_listt,
                            shareclass=shareclass,
                            printlog=True,
                            corrmethod=corrmethod,
                            reb_period=reb_period,
                            lookback_period_short=lookback_period_short,
                            lookback_period_long=lookback_period_long,
                            moving_average_period=moving_average_period,
                            momentum_period=momentum_period,
                            momentum_percentile=momentum_percentile,
                            leverage=params['leverage']
                            )

    # otherwise, rely on the weights of a strategy
    else:
        cerebro.addstrategy(eval(strategy),
                            n_assets=n_assets,
                            shares=shares_list,
                            initial_cash=params['initial_cash'],
                            contribution=params['contribution'],
                            shareclass=shareclass,
                            printlog=True,
                            corrmethod=corrmethod,
                            reb_period=reb_period,
                            lookback_period_short=lookback_period_short,
                            lookback_period_long=lookback_period_long,
                            moving_average_period=moving_average_period,
                            momentum_period=momentum_period,
                            momentum_percentile=momentum_percentile,
                            leverage=params['leverage']
                            )

    # Run backtest
    cerebro.run()
    #cerebro.plot(volume=False)

    # Create report
    if params['create_report']:
        OutputList = cerebro.report(system=platform.system())

        return OutputList

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def main(params):
    # Fund mode if contribution is 0 otherwise, asset mode
    if params['initial_cash'] == '':
        return False

    if params['contribution'] == 0:
        params["fundmode"] = True
    else:
        params["fundmode"] = False

    strategy_list = []
    if params['riskparity']:
        strategy_list.append('riskparity')
    if params['riskparity_nested']:
        strategy_list.append('riskparity_nested')
    if params['rotationstrat']:
        strategy_list.append('rotationstrat')
    if params['uniform']:
        strategy_list.append('uniform')
    if params['vanillariskparity']:
        strategy_list.append('vanillariskparity')
    if params['onlystocks']:
        strategy_list.append('onlystocks')
    if params['sixtyforty']:
        strategy_list.append('sixtyforty')
    if params['trend_u']:
        strategy_list.append('trend_u')
    if params['absmom_u']:
        strategy_list.append('absmom_u')
    if params['relmom_u']:
        strategy_list.append('relmom_u')
    if params['momtrend_u']:
        strategy_list.append('momtrend_u')
    if params['trend_rp']:
        strategy_list.append('trend_rp')
    if params['absmom_rp']:
        strategy_list.append('absmom_rp')
    if params['relmom_rp']:
        strategy_list.append('relmom_rp')
    if params['momtrend_rp']:
        strategy_list.append('momtrend_rp')
    if params['GEM']:
        strategy_list.append('GEM')
    if params['acc_dualmom']:
        strategy_list.append('acc_dualmom')
    if params['acc_dualmom2']:
        strategy_list.append('acc_dualmom2')
    if params['rot_adm']:
        strategy_list.append('rot_adm')
    if params['rot_adm_dual_rp']:
        strategy_list.append('rot_adm_dual_rp')
    if params['vigilant']:
        strategy_list.append('vigilant')
    if params['TESTING']:
        strategy_list.append('TESTING')
    if params['specific_vigilant']:
        strategy_list.append('specific_vigilant')
    if params['specific_rot']:
        strategy_list.append('specific_rot')
    if params['specific_rot2']:
        strategy_list.append('specific_rot2')
    if params['specific_adm']:
        strategy_list.append('specific_adm')
    if params['specific_adm_grad_div']:
        strategy_list.append('specific_adm_grad_div')
    if params['specific_fabio_adm2']:
        strategy_list.append('specific_fabio_adm2')
    if params['specific_fabiofg_adm2']:
        strategy_list.append('specific_fabiofg_adm2')
    if params['specific_fabio_adm3']:
        strategy_list.append('specific_fabio_adm3')
    if params['specific']:
        strategy_list.append('specific')
    if not strategy_list:
        strategy_list = ["customweights"]
    if params['benchmark'] != '':
        strategy_list = strategy_list + ['benchmark']

    print_header(params,strategy_list)

    # Output list description:
    # list index, content
    # 0, prices
    # 1, returns
    # 2, performance data
    # 3, target weights
    # 4, effective weights
    # 5, portfolio drawdown
    # 6, assetprices
    # 7, assets drawdown
    # 8, parameters
    InputList = []
    stratIndependentOutput = [6, 7, 8] # these indexes correspond to the strategy independent outputs

    for i in range(0,9):
        InputList.append(pd.DataFrame())

    for strat in strategy_list:
        print_section_divider(strat)
        st.write("Backtesting strategy: " + strat)

        ThisOutputList = runOneStrat(strat)

        for i in range(0, len(ThisOutputList)):
            if strat == strategy_list[0] or i in stratIndependentOutput:
                InputList[i] = ThisOutputList[i]
            elif i in [3,4]:
                InputList[i] = InputList[i].append(ThisOutputList[i])
            else:
                InputList[i][strat] = ThisOutputList[i]

    if params['report_name'] != '':
        outfilename = params['report_name'] + "_" + get_now() + ".html"
    else:
        outfilename = "Report_" + get_now() + ".html"

    ReportAggregator = report_aggregator.ReportAggregator(outfilename, params['user'], params['memo'], params['leverage'], platform.system(), InputList)
    #ReportAggregator.report()
    OutputList = ReportAggregator.report_object()
    return OutputList
