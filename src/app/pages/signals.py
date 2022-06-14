import streamlit as st
from scipy import stats
import os, sys
import plotly.express as px
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image
import riskparityportfolio as rp
import math
# from pages.home import session_state


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import SessionState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils

def app():
    st.title('Market Signals')

    st.write('Useful market signals from popular strategies.')

    st.markdown("## Rotational strategy")
    st.write("Signal for the asset rotation strategy that buys either gold, bonds or equities"
             " (see https://seekingalpha.com/article/4283733-simple-rules-based-asset-rotation-strategy).")

    rotstrat_curves = utils.load_economic_curves(date.today()-timedelta(365*2), date.today())
    rotstrat_curves = rotstrat_curves.drop(['Max'],axis=1)
    columns = rotstrat_curves.columns
    rotstrat_curves['date'] = rotstrat_curves.index
    rotstrat_curves_long = pd.melt(rotstrat_curves, id_vars=['date'], value_vars=columns, var_name='asset', value_name='signal')

    fig = px.line(rotstrat_curves_long, x="date", y="signal", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    end_month = rotstrat_curves.groupby(rotstrat_curves.index.month).max().sort_values('date').iloc[-2]['date']


    st.markdown("- If the yield curve (T10Y2Y) is highest, buy stocks. ")
    st.markdown("- If the inflation expectation rate minus the yield curve (T10YIE_T10Y2Y) is highest, buy gold.")
    st.markdown("- If the 20-year TIP rate (DFII20) is highest, buy long-term bonds.")

    RotStrat_stocks = [['DFII20', '20-Year Treasury Inflation-Indexed Security, Constant Maturity', rotstrat_curves.loc[(rotstrat_curves.index == end_month)]['DFII20'].values[0], rotstrat_curves.loc[(rotstrat_curves.index == rotstrat_curves.index.max())]['DFII20'].values[0], 'https://fred.stlouisfed.org/series/DFII20'],
                      ['T10Y2Y', '10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity', rotstrat_curves.loc[(rotstrat_curves.index == end_month)]['T10Y2Y'].values[0], rotstrat_curves.loc[(rotstrat_curves.index == rotstrat_curves.index.max())]['T10Y2Y'].values[0], 'https://fred.stlouisfed.org/series/T10Y2Y'],
                  ['T10YIE_T10Y2Y', '10-Year Breakeven Inflation Rate minus T10Y2Y',rotstrat_curves.loc[(rotstrat_curves.index == end_month)]['T10YIE_T10Y2Y'].values[0], rotstrat_curves.loc[(rotstrat_curves.index == rotstrat_curves.index.max())]['T10YIE_T10Y2Y'].values[0], 'https://fred.stlouisfed.org/series/T10YIE']]

    RotStrat_stocks_df = pd.DataFrame(RotStrat_stocks, columns=['ticker', 'name', 'month end score','today score','link'])
    st.dataframe(RotStrat_stocks_df)

    RotStrat_stocks_df_eom = RotStrat_stocks_df[['ticker', 'month end score']]
    RotStrat_stocks_df_today = RotStrat_stocks_df[['ticker', 'today score']]

    ticker_eom=RotStrat_stocks_df_eom.iloc[RotStrat_stocks_df_eom['month end score'].idxmax()]['ticker']
    ticker_today=RotStrat_stocks_df_eom.iloc[RotStrat_stocks_df_today['today score'].idxmax()]['ticker']

    if ticker_eom == 'DFII20':
        ticker_eom = "long-term bonds"
    elif ticker_eom == 'T10Y2Y':
        ticker_eom = "stocks"
    elif ticker_eom == 'T10YIE_T10Y2Y':
        ticker_eom = "gold"

    if ticker_today == 'DFII20':
        ticker_today = "long-term bonds"
    elif ticker_today == 'T10Y2Y':
        ticker_today = "stocks"
    elif ticker_today == 'T10YIE_T10Y2Y':
        ticker_today = "gold"

    st.markdown('The asset allocation recommended by the rotational strategy is: **' + ticker_eom + '**, based  on last month ('+ end_month.strftime("%m/%Y") +') returns.')
    st.markdown("Today (" + rotstrat_curves.index.max().strftime("%d/%m/%Y") + "), the asset allocation recommended by the rotational strategy is: **" + ticker_today + " **.")

    rotstrat_curves_long_now = rotstrat_curves_long.where(rotstrat_curves_long.date == rotstrat_curves_long.date.max())
    rot_asset = rotstrat_curves_long_now.iloc[rotstrat_curves_long_now['signal'].idxmax(axis=0, skipna=True)]['asset']
    rotstrat_curves_long_now['asset_weight'] = 0
    rotstrat_curves_long_now.loc[rotstrat_curves_long_now['asset'] == rot_asset, 'asset_weight'] = 1

    rotstrat_curves_long_now['ticker'] = ''
    rotstrat_curves_long_now.loc[rotstrat_curves_long_now['asset'] == 'DFII20', 'ticker'] = 'TLT'
    rotstrat_curves_long_now.loc[rotstrat_curves_long_now['asset'] == 'T10Y2Y', 'ticker'] = 'SPY'
    rotstrat_curves_long_now.loc[rotstrat_curves_long_now['asset'] == 'T10YIE_T10Y2Y', 'ticker'] = 'GLD'
    rotstrat_curves_long_now.drop(columns = ['date','asset','signal'])

    st.markdown("## Risk Parity")
    st.write("Dynamic allocation of weights according to the risk parity methodology "
             "(see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/)."
             "Example implemented with 5 assets of classes equity, bond_lt, bond_it, gold, commodity: "
             "VOO, TLT, IEF, GLD, GSG."
             )

    st.write("Using 120 days to estimate the asset correlation and 20 days for the volatility")

    shares_list = ['VOO', 'TLT', 'IEF', 'GLD', 'GSG']
    RP_data = utils.load_riskparity_data(shares_list, date.today()-timedelta(365*3), date.today())

    n_assets = 5
    lookback_period_long = 120
    lookback_period_short = 20
    corrmethod = 'pearson'

    target_risk = [1 / 5] * 5  # Same risk for each asset = risk parity
    RP_data['Date'] = RP_data.index
    RP_data_pivot = RP_data.pivot(index='Date', columns='asset', values='close')
    RP_data_pivot_prices = RP_data_pivot.sort_index(ascending=True).tail(lookback_period_long)
    logrets = np.diff(np.log(RP_data_pivot_prices), axis=0)

    if corrmethod == 'pearson':
        corr = np.corrcoef(logrets, rowvar=False)
    elif corrmethod == 'spearman':
        corr, p, = stats.spearmanr(logrets, axis=0)

    stddev = np.array(
        np.std(logrets[len(logrets) - lookback_period_short:len(logrets)],axis=0))  # standard dev indicator
    stddev_matrix = np.diag(stddev)
    cov = stddev_matrix @ corr @ stddev_matrix  # covariance matrix

    weights = rp.RiskParityPortfolio(covariance=cov, budget=target_risk).weights

    rp_stocks = [['equity',shares_list[0], 'Vanguard S&P 500 ETF', 'https://finance.yahoo.com/quote/VOO'],
                  ['bond_lt',shares_list[1], 'iShares 20+ Year Treasury Bond ETF', 'https://finance.yahoo.com/quote/TLT'],
                  ['bond_lt',shares_list[2], 'iShares 7-10 Year Treasury Bond ETF', 'https://finance.yahoo.com/quote/IEF'],
                 ['gold',shares_list[3], 'SPDR Gold Shares', 'https://finance.yahoo.com/quote/GLD'],
                 ['commodity',shares_list[4], 'iShares S&P GSCI Commodity-Indexed Trust', 'https://finance.yahoo.com/quote/GSG']]
    rp_stocks_df = pd.DataFrame(rp_stocks, columns=['type','ticker', 'name', 'link'])
    rp_stocks_df["weights"] = weights

    rp_stocks_df = rp_stocks_df.style.format({'weights': "{:.2%}"})

    st.dataframe(rp_stocks_df)


    st.markdown("## GEM")
    st.write("Global equity momentum strategy. Needs only 4 assets of classes equity, equity_intl, bond_lt, money_market. "
             "example: `VEU,IVV,BIL,AGG equity_intl,equity,money_market,bond_lt`. "
             "See https://blog.thinknewfound.com/2019/01/fragility-case-study-dual-momentum-gem/")

    shares_list = ['VEU','IVV','BIL']
    GEM_curves = utils.load_GEM_curves(shares_list, date.today()-timedelta(365*3), date.today())
    fig = px.line(GEM_curves, x="date", y="return", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    end_month = GEM_curves.groupby(GEM_curves.index.month).max().sort_values('date').iloc[-2]['date']

    wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    GEM_image_path = utils.find('GEM.png', wd)
    GEM_image = Image.open(GEM_image_path)
    st.image(GEM_image)

    GEM_stocks = [['equity_intl',shares_list[0], 'Vanguard FTSE All-World ex-US Index Fund ETF Shares ', GEM_curves.loc[(GEM_curves.asset == shares_list[0]) & (GEM_curves.index == end_month)]['return'].values[0], GEM_curves.loc[(GEM_curves.asset == shares_list[0]) & (GEM_curves.index == GEM_curves.index.max())]['return'].values[0], 'https://finance.yahoo.com/quote/VEU'],
                  ['equity',shares_list[1], 'iShares Core S&P 500 ETF', GEM_curves.loc[(GEM_curves.asset == shares_list[1]) & (GEM_curves.index == end_month)]['return'].values[0], GEM_curves.loc[(GEM_curves.asset == shares_list[1]) & (GEM_curves.index == GEM_curves.index.max())]['return'].values[0], 'https://finance.yahoo.com/quote/IVV'],
                  ['money_market',shares_list[2], 'SPDR Bloomberg Barclays 1-3 Month T-Bill ETF', GEM_curves.loc[(GEM_curves.asset == shares_list[2]) & (GEM_curves.index == end_month)]['return'].values[0], GEM_curves.loc[(GEM_curves.asset == shares_list[2]) & (GEM_curves.index == GEM_curves.index.max())]['return'].values[0],'https://finance.yahoo.com/quote/BIL'],
                  ['bond_lt',"AGG", 'iShares Core U.S. Aggregate Bond ETF (AGG)', math.nan, math.nan,'https://finance.yahoo.com/quote/AGG']]
    GEM_stocks_df = pd.DataFrame(GEM_stocks, columns=['type','ticker', 'name', 'month end score','today score','link'])
    st.dataframe(GEM_stocks_df)

    score_col_name = 'month end score'
    data = GEM_stocks_df

    if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
            data.loc[data['type'] == 'money_market', score_col_name].values[0]:
        if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
                data.loc[data['type'] == 'equity_intl', score_col_name].values[0]:
            ticker_eom = data.loc[data['type'] == 'equity', 'ticker'].values[0]
        else:
            ticker_eom = data.loc[data['type'] == 'equity_intl', 'ticker'].values[0]
    else:
        ticker_eom = "AGG"

    st.markdown("The Global Equity Momentum allocation recommends to allocate resources to **" + ticker_eom + "**, based  on last month ("+ end_month.strftime("%m/%Y") + ") returns.")

    score_col_name = 'today score'
    data = GEM_stocks_df

    if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
            data.loc[data['type'] == 'money_market', score_col_name].values[0]:
        if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
                data.loc[data['type'] == 'equity_intl', score_col_name].values[0]:
            ticker_eom = data.loc[data['type'] == 'equity', 'ticker'].values[0]
        else:
            ticker_eom = data.loc[data['type'] == 'equity_intl', 'ticker'].values[0]
    else:
        ticker_eom = "AGG"

    st.markdown("Today (" + GEM_curves.index.max().strftime("%d/%m/%Y") + "), the Global Equity Momentum allocation would recommend to allocate resources to **" + ticker_eom + "**.")


    st.markdown("## Accelerating dual momentum")
    st.write("Accelerating Dual Momentum. Needs only 3 assets of classes equity, equity_intl, bond_lt. example: "
             "VFINX,VINEX,VUSTX, shareclass equity,equity_intl,bond_lt. "
             "See https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/")

    shares_list = ['VFINX','VINEX','VUSTX']
    # shares_list = ['SPY','SCZ','TLT']
    AccDualMom_curves = utils.load_AccDualMom_curves(shares_list, date.today()-timedelta(365*2.5), date.today())

    # add the 3m t bill
    t_bill = utils.load_fred_curve(date.today()-timedelta(365*2.5), date.today(), ['DTB3'])
    t_bill = t_bill.rename(columns={"DTB3": "score"})
    t_bill['score'] = t_bill['score']/100
    t_bill['asset'] = '3m_tbill'
    #t_bill['date'] = t_bill.index
    t_bill = pd.merge(t_bill, AccDualMom_curves['date'], how='right', left_index=True, right_index=True)
    t_bill = t_bill.fillna(method="ffill")
    t_bill = t_bill.fillna(method="bfill")
    AccDualMom_curves = AccDualMom_curves.append(t_bill)

    fig = px.line(AccDualMom_curves, x="date", y="score", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    end_month = AccDualMom_curves.groupby(AccDualMom_curves.index.month).max().sort_values('date').iloc[-2]['date']

    col1, col2 = st.columns(2)
    wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    AccDualMom_image_path = utils.find('acc_dualmom.png', wd)
    AccDualMom_image = Image.open(AccDualMom_image_path)
    col1.image(AccDualMom_image)

    AccDualMom_stocks = [['equity',shares_list[0], 'Vanguard 500 Index Fund Investor Shares', AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[0]) & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[0]) & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/VFINX'],
                  ['equity_intl',shares_list[1], 'Vanguard International Explorer Fund Investor Shares', AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[1]) & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[1]) & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0],  'https://finance.yahoo.com/quote/VINEX'],
                  ['3m_bills','DTB3', '3-Month Treasury Bill Secondary Market Rate', AccDualMom_curves.loc[(AccDualMom_curves.asset == '3m_tbill') & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == '3m_tbill') & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0], 'https://fred.stlouisfed.org/series/DTB3']]
    AccDualMom_stocks_df = pd.DataFrame(AccDualMom_stocks, columns=['type','ticker', 'name', 'month end score','today score','link'])
    st.dataframe(AccDualMom_stocks_df)

    score_col_name = 'month end score'
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

    st.markdown("The accelerated dual momentum allocation recommends to allocate resources to **" + ticker_eom + "**, based  on last month (" + end_month.strftime("%m/%Y") + ")  returns.")

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
    st.markdown("Today (" + AccDualMom_curves.index.max().strftime("%d/%m/%Y") + "), the accelerated dual momentum allocation would recommend to allocate resources to **" + ticker_eom + "**.")

    # TODO use common functions for ADM signals
    st.markdown("## Accelerating dual momentum")
    st.write("Accelerating Dual Momentum. Needs only 3 assets of classes equity, equity_intl, bond_lt. example: "
             "VFINX,VINEX,VUSTX, shareclass equity,equity_intl,bond_lt. "
             "See https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/")

    # shares_list = ['VFINX','VINEX','VUSTX']
    shares_list = ['VFINX', 'VINEX', 'TLT', 'IEF', 'GLD', 'GSG']
    # shares_list = ['SPY','SCZ','TLT']
    AccDualMom_curves = utils.load_AccDualMom_curves(shares_list, date.today()-timedelta(365*2.5), date.today())

    # add the 3m t bill
    t_bill = utils.load_fred_curve(date.today()-timedelta(365*2.5), date.today(), ['DTB3'])
    t_bill = t_bill.rename(columns={"DTB3": "score"})
    t_bill['score'] = t_bill['score']/100
    t_bill['asset'] = '3m_tbill'
    #t_bill['date'] = t_bill.index
    t_bill = pd.merge(t_bill, AccDualMom_curves['date'], how='right', left_index=True, right_index=True)
    t_bill = t_bill.fillna(method="ffill")
    t_bill = t_bill.fillna(method="bfill")
    AccDualMom_curves = AccDualMom_curves.append(t_bill)

    fig = px.line(AccDualMom_curves, x="date", y="score", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    end_month = AccDualMom_curves.groupby(AccDualMom_curves.index.month).max().sort_values('date').iloc[-2]['date']

    col1, col2 = st.columns(2)
    wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    AccDualMom_image_path = utils.find('acc_dualmom.png', wd)
    AccDualMom_image = Image.open(AccDualMom_image_path)
    col1.image(AccDualMom_image)

    AccDualMom_stocks = [['equity',shares_list[0], 'Vanguard 500 Index Fund Investor Shares', AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[0]) & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[0]) & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/VFINX'],
                  ['equity_intl',shares_list[1], 'Vanguard International Explorer Fund Investor Shares', AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[1]) & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == shares_list[1]) & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0],  'https://finance.yahoo.com/quote/VINEX'],
                  ['3m_bills','DTB3', '3-Month Treasury Bill Secondary Market Rate', AccDualMom_curves.loc[(AccDualMom_curves.asset == '3m_tbill') & (AccDualMom_curves.index == end_month)]['score'].values[0], AccDualMom_curves.loc[(AccDualMom_curves.asset == '3m_tbill') & (AccDualMom_curves.index == AccDualMom_curves.index.max())]['score'].values[0], 'https://fred.stlouisfed.org/series/DTB3']]
    AccDualMom_stocks_df = pd.DataFrame(AccDualMom_stocks, columns=['type','ticker', 'name', 'month end score','today score','link'])
    st.dataframe(AccDualMom_stocks_df)

    score_col_name = 'month end score'
    data = AccDualMom_stocks_df

    utils.ADM_weights(data, end_month)

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

    st.markdown("The accelerated dual momentum allocation recommends to allocate resources to **" + ticker_eom + "**, based  on last month (" + end_month.strftime("%m/%Y") + ")  returns.")

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
    st.markdown("Today (" + AccDualMom_curves.index.max().strftime("%d/%m/%Y") + "), the accelerated dual momentum allocation would recommend to allocate resources to **" + ticker_eom + "**.")
    # END TODO



    st.markdown("## Accelerating dual momentum (with GLD, Commodities)")
    st.write("Accelerating Dual Momentum, including TIPs as inflation hedge. Needs only 4 assets of classes equity, "
             "equity_intl, bond_lt, gold. example: "
             "VFINX,VINEX,VUSTX,GLD,GSG  shareclass equity, equity_intl, bond_lt, gold, commodity. "
             "See https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/")

    shares_list = ['VFINX','VINEX','VUSTX','GLD','GSG']
    AccDualMom_curves2 = utils.load_AccDualMom_curves(shares_list, date.today()-timedelta(365*2.5), date.today())
    fig = px.line(AccDualMom_curves2, x="date", y="score", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    #col1, col2 = st.columns(2)
    #wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    #AccDualMom_image_path = utils.find('acc_dualmom.png', wd)
    #AccDualMom_image = Image.open(AccDualMom_image_path)
    #col1.image(AccDualMom_image)

    end_month = AccDualMom_curves2.groupby(AccDualMom_curves2.index.month).max().sort_values('date').iloc[-2]['date']

    AccDualMom2_stocks =  [['equity',shares_list[0], 'Vanguard 500 Index Fund Investor Shares', AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[0]) & (AccDualMom_curves2.index == end_month)]['score'].values[0], AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[0]) & (AccDualMom_curves2.index == AccDualMom_curves2.index.max())]['score'].values[0], 'https://finance.yahoo.com/quote/VFINX'],
                          ['equity_intl',shares_list[1], 'Vanguard International Explorer Fund Investor Shares', AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[1]) & (AccDualMom_curves2.index == end_month)]['score'].values[0], AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[1]) & (AccDualMom_curves2.index == AccDualMom_curves2.index.max())]['score'].values[0], 'https://finance.yahoo.com/quote/VINEX'],
                          ['bond_lt',shares_list[2], 'Vanguard Long-Term Treasury Fund Investor Shares',AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[2]) & (AccDualMom_curves2.index == end_month)]['score'].values[0], AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[2]) & (AccDualMom_curves2.index == AccDualMom_curves2.index.max())]['score'].values[0], 'https://finance.yahoo.com/quote/VUSTX'],
                          ['gold',shares_list[3], 'SPDR Gold Shares (GLD)',AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[3]) & (AccDualMom_curves2.index == end_month)]['score'].values[0], AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[3]) & (AccDualMom_curves2.index == AccDualMom_curves2.index.max())]['score'].values[0], 'https://finance.yahoo.com/quote/GLD'],
                          ['commodity',shares_list[4], 'iShares S&P GSCI Commodity-Indexed Trust (GSG)', AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[4]) & (AccDualMom_curves2.index == end_month)]['score'].values[0], AccDualMom_curves2.loc[(AccDualMom_curves2.asset == shares_list[4]) & (AccDualMom_curves2.index == AccDualMom_curves2.index.max())]['score'].values[0], 'https://finance.yahoo.com/quote/gsg'],
                         ]
    AccDualMom2_stocks_df = pd.DataFrame(AccDualMom2_stocks, columns=['type','ticker', 'name', 'month end score','today score','link'])
    #col2.dataframe(AccDualMom_stocks_df)
    st.dataframe(AccDualMom2_stocks_df)

    score_col_name = 'month end score'
    data = AccDualMom2_stocks_df
    if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
            data.loc[data['type'] == 'equity_intl', score_col_name].values[0]:
        if data.loc[data['type'] == 'equity', score_col_name].values[0] > 0:
            ticker_eom = data.loc[data['type'] == 'equity', 'ticker'].values[0]
        else:
            if data.loc[data['type'] == 'bond_lt', score_col_name].values[0] > \
                    data.loc[data['type'] == 'gold', score_col_name].values[0]:
                ticker_eom = data.loc[data['type'] == 'bond_lt', 'ticker'].values[0]
            else:
                ticker_eom = data.loc[data['type'] == 'gold', 'ticker'].values[0]
    else:
        if data.loc[data['type'] == 'equity_intl', score_col_name].values[0] > 0:
            ticker_eom = data.loc[data['type'] == 'equity_intl', 'ticker'].values[0]
        else:
            if data.loc[data['type'] == 'bond_lt', score_col_name].values[0] > \
                    data.loc[data['type'] == 'gold', score_col_name].values[0]:
                ticker_eom = data.loc[data['type'] == 'bond_lt', 'ticker'].values[0]
            else:
                ticker_eom = data.loc[data['type'] == 'gold', 'ticker'].values[0]

    st.markdown("The accelerated dual momentum (with gold) allocation recommends to allocate resources to **" + ticker_eom + "**, based  on last month  ("+ end_month.strftime("%m/%Y") +")  returns.")

    score_col_name = 'today score'
    data = AccDualMom2_stocks_df
    if data.loc[data['type'] == 'equity', score_col_name].values[0] > \
            data.loc[data['type'] == 'equity_intl', score_col_name].values[0]:
        if data.loc[data['type'] == 'equity', score_col_name].values[0] > 0:
            ticker_eom = data.loc[data['type'] == 'equity', 'ticker'].values[0]
        else:
            if data.loc[data['type'] == 'bond_lt', score_col_name].values[0] > \
                    data.loc[data['type'] == 'gold', score_col_name].values[0]:
                ticker_eom = data.loc[data['type'] == 'bond_lt', 'ticker'].values[0]
            else:
                ticker_eom = data.loc[data['type'] == 'gold', 'ticker'].values[0]
    else:
        if data.loc[data['type'] == 'equity_intl', score_col_name].values[0] > 0:
            ticker_eom = data.loc[data['type'] == 'equity_intl', 'ticker'].values[0]
        else:
            if data.loc[data['type'] == 'bond_lt', score_col_name].values[0] > \
                    data.loc[data['type'] == 'gold', score_col_name].values[0]:
                ticker_eom = data.loc[data['type'] == 'bond_lt', 'ticker'].values[0]
            else:
                ticker_eom = data.loc[data['type'] == 'gold', 'ticker'].values[0]
    st.markdown("Today (" + AccDualMom_curves2.index.max().strftime("%d/%m/%Y") + "), the accelerated dual momentum (with gold) allocation would recommend to allocate resources to **" + ticker_eom + "**.")


    st.markdown("## Vigilant asset allocation")
    st.write("Vigilant asset allocation from https://allocatesmartly.com/vigilant-asset-allocation-dr-wouter-keller-jw-keuning/."
             " Example: SPY,EFA,EEM,AGG,LQD,IEF,SHY shareclass equity,equity_intl,equity_intl,bond_lt,bond_lt,bond_it,bond_it. ")

    shares_list = ['SPY','EFA','FEMKX','AGG','LQD','IEF','SHY']
    vigilant_curves = utils.load_vigilant_curves(shares_list, date.today()-timedelta(365*2.5), date.today())
    fig = px.line(vigilant_curves, x="date", y="score", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    end_month = vigilant_curves.groupby(vigilant_curves.index.month).max().sort_values('date').iloc[-2]['date']

    vigilant_stocks =  [['offensive',shares_list[0],'SPDR S&P 500 ETF Trust (SPY)',vigilant_curves.loc[(vigilant_curves.asset == shares_list[0]) & (vigilant_curves.index == end_month)]['score'].values[0],vigilant_curves.loc[(vigilant_curves.asset == shares_list[0]) & (vigilant_curves.index == vigilant_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/SPY'],
                        ['offensive',shares_list[1],'iShares MSCI EAFE ETF (EFA)',vigilant_curves.loc[(vigilant_curves.asset == shares_list[1]) & (vigilant_curves.index == end_month)]['score'].values[0],vigilant_curves.loc[(vigilant_curves.asset == shares_list[1]) & (vigilant_curves.index == vigilant_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/EFA'],
                        ['offensive',shares_list[2],'Fidelity Emerging Markets (FEMKX)',vigilant_curves.loc[(vigilant_curves.asset == shares_list[2]) & (vigilant_curves.index == end_month)]['score'].values[0],vigilant_curves.loc[(vigilant_curves.asset == shares_list[2]) & (vigilant_curves.index == vigilant_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/FEMKX'],
                        ['offensive',shares_list[3],'iShares Core U.S. Aggregate Bond ETF (AGG)',vigilant_curves.loc[(vigilant_curves.asset == shares_list[3]) & (vigilant_curves.index == end_month)]['score'].values[0],vigilant_curves.loc[(vigilant_curves.asset == shares_list[3]) & (vigilant_curves.index == vigilant_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/AGG'],
                        ['defensive',shares_list[4],'iShares iBoxx $ Investment Grade Corporate Bond ETF (LQD)',vigilant_curves.loc[(vigilant_curves.asset == shares_list[4]) & (vigilant_curves.index == end_month)]['score'].values[0], vigilant_curves.loc[(vigilant_curves.asset == shares_list[4]) & (vigilant_curves.index == vigilant_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/LQD'],
                        ['defensive',shares_list[5],'iShares 7-10 Year Treasury Bond ETF (IEF)',vigilant_curves.loc[(vigilant_curves.asset == shares_list[5]) & (vigilant_curves.index == end_month)]['score'].values[0],vigilant_curves.loc[(vigilant_curves.asset == shares_list[5]) & (vigilant_curves.index == vigilant_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/IEF'],
                        ['defensive',shares_list[6],'iShares 1-3 Year Treasury Bond ETF (SHY)',vigilant_curves.loc[(vigilant_curves.asset == shares_list[6]) & (vigilant_curves.index == end_month)]['score'].values[0],vigilant_curves.loc[(vigilant_curves.asset == shares_list[6]) & (vigilant_curves.index == vigilant_curves.index.max())]['score'].values[0],'https://finance.yahoo.com/quote/SHY']
                       ]
    vigilant_stocks_df = pd.DataFrame(vigilant_stocks, columns=['type','ticker','name','month end score','today score', 'link'])
    #col2.dataframe(AccDualMom_stocks_df)
    st.dataframe(vigilant_stocks_df)

    score_col_name = 'month end score'
    data = vigilant_stocks_df

    if all(data.loc[data['type'] == 'offensive', score_col_name] > 0):
        momentum_df_off = data.where(data.type == "offensive")
        ticker_eom = momentum_df_off.iloc[momentum_df_off[score_col_name].idxmax(axis=0, skipna=True)]['ticker']
    else:
        momentum_df_def = data.where(data.type == "defensive")
        ticker_eom = momentum_df_def.iloc[momentum_df_def[score_col_name].idxmax(axis=0, skipna=True)]['ticker']

    st.markdown("The vigilant asset allocation recommends to allocate resources to **" + ticker_eom + "**, based  on last month  ("+ end_month.strftime("%m/%Y") +")  returns.")

    score_col_name = 'today score'
    data = vigilant_stocks_df

    if all(data.loc[data['type'] == 'offensive', score_col_name] > 0):
        momentum_df_off = data.where(data.type == "offensive")
        ticker_eom = momentum_df_off.iloc[momentum_df_off[score_col_name].idxmax(axis=0, skipna=True)]['ticker']
    else:
        momentum_df_def = data.where(data.type == "defensive")
        ticker_eom = momentum_df_def.iloc[momentum_df_def[score_col_name].idxmax(axis=0, skipna=True)]['ticker']

    st.markdown("Today (" + vigilant_curves.index.max().strftime("%d/%m/%Y") + "), the vigilant asset allocation would recommend to allocate resources to **" + ticker_eom + "**.")
