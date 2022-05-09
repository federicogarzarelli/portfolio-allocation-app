import copy
import streamlit as st
from datetime import date, datetime, timedelta
from scipy import stats
import os, sys
import plotly.express as px
import pandas as pd
import numpy as np
import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import SessionState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from main import main
from GLOBAL_VARS import params

# session_state = SessionState.get(ib_client = None,
#                                  live_backtest=False,
#                                  startdate=datetime.strptime('2000-01-01', '%Y-%m-%d'), enddate = date.today(),
#                                  initial_cash=1000000.0, contribution=0.0, leverage=0.0, expense_ratio=0.01,
#                                  historic="Yahoo Finance (daily prices)", #"Historical DB (daily prices)",
#                                  # shares=['VFINX', 'VINEX', 'TLT', 'IEF', 'GLD', 'GSG', 'QQQ', 'TIP', 'EFA', 'EEM','AGG','LQD','SHY','BRK-B',''],
#                                  # shareclass=['equity','equity_intl','bond_lt','bond_it','gold','commodity','equity','bond_lt','equity','equity','bond_lt','bond_lt','bond_it','equity',''],
#                                  # FABIO'S strats
#                                 shares=['SPY','EFA','FEMKX','AGG','LQD','IEF','SHY','VFINX','VINEX','VUSTX','QQQ','MDY','FXI','VNQ','GLD','TLT','VBMFX', 'BRK-B', 'TIP', 'GSG'],
#                                 shareclass=['equity','equity_intl','equity_intl','bond_lt','bond_lt','bond_it','bond_it','equity','equity_intl','bond_lt','equity','equity','equity_intl','equity','gold','bond_lt','bond_lt', 'equity', 'bond_lt', 'commodity'],
#                                  weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                  # my strats
#                                  # shares=   ['SPY','QQQ','MDY','EFA','FXI','FEMKX','VNQ','GLD','TLT','LQD','VBMFX', '', '', '', ''],
#                                  # shareclass=['equity','equity','equity','equity_intl','equity_intl','equity_intl','equity','gold','bond_lt','bond_lt','bond_lt','','','',''],
#                                  # weights=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
#                                  benchmark='',
#                                  riskparity=False, riskparity_nested=False, rotationstrat=False, uniform=False, vanillariskparity=False, onlystocks=False, sixtyforty=False,
#                                  trend_u=False, absmom_u=False, relmom_u=False, momtrend_u=False, trend_rp=False, absmom_rp=False, relmom_rp=False, momtrend_rp=False, GEM=False,
#                                  acc_dualmom=False, acc_dualmom2=False, rot_adm=False,rot_adm_dual_rp=False, TESTING=True, vigilant=False,
#                                  specific_vigilant=False,
#                                  specific_rot=False,
#                                  specific_rot2=False,
#                                  specific_adm=False,
#                                  specific_adm_grad_div=False,
#                                  specific_fabio_adm2=False,
#                                  specific_fabiofg_adm2=False,
#                                  specific_fabio_adm3=False,
#                                  specific=False,
#
#                                  create_report=True, report_name='backtest report', user='FG', memo='backtest report',
#                                  # advanced parameters
#                                  MARGINLOAN_INT_RATE=1.5,
#                                  DAYS_IN_YEAR=252, DAYS_IN_YEAR_BOND_PRICE=360,
#                                  reb_period_days=1,reb_period_years=1,reb_period_custweights=1,
#                                  lookback_period_short_days=20, lookback_period_short_years=5, lookback_period_short_custweights=20,
#                                  lookback_period_long_days=120, lookback_period_long_years=10, lookback_period_long_custweights=120,
#                                  moving_average_period_days=252,moving_average_period_years=5, moving_average_period_custweights=252,
#                                  momentum_period_days=252, momentum_period_years=5, momentum_period_custweights=252,
#                                  momentum_percentile_days=0.5, momentum_percentile_years=0.5,
#                                  momentum_percentile_custweights=0.5,
#                                  corrmethod_days='pearson', corrmethod_years='pearson',
#                                  corrmethod_custweights='pearson',
#                                  riskfree=0.01, targetrate=0.01, alpha=0.05, market_mu=0.07, market_sigma=0.15,
#                                  stddev_sample=True, annualize=True, logreturns=False,
#                                  assets_startdate=datetime.strptime('2010-01-01', '%Y-%m-%d'),assets_enddate = datetime.strptime('2021-01-01', '%Y-%m-%d'),
#                                  assets_multiselect=['SP500','ZB.F','ZN.F','BM.F','GC.C'],
#                                  userallocation_live=False,
#                                  ADM_live=False)
st.session_state.ib_client = None
st.session_state.live_backtest=False
st.session_state.startdate=datetime.strptime('2000-01-01', '%Y-%m-%d')
st.session_state.enddate = date.today()
st.session_state.initial_cash=1000000.0
st.session_state.contribution=0.0
st.session_state.leverage=0.0
st.session_state.expense_ratio=0.01
st.session_state.historic="Yahoo Finance (daily prices)", #"Historical DB (daily prices)"
# st.session_state.shares=['VFINX', 'VINEX', 'TLT', 'IEF', 'GLD', 'GSG', 'QQQ', 'TIP', 'EFA', 'EEM','AGG','LQD','SHY','BRK-B','']
# st.session_state.shareclass=['equity','equity_intl','bond_lt','bond_it','gold','commodity','equity','bond_lt','equity','equity','bond_lt','bond_lt','bond_it','equity','']
# FABIO'S strats
st.session_state.shares=['SPY','EFA','FEMKX','AGG','LQD','IEF','SHY','VFINX','VINEX','VUSTX','QQQ','MDY','FXI','VNQ','GLD','TLT','VBMFX', 'BRK-B', 'TIP', 'GSG']
st.session_state.shareclass=['equity','equity_intl','equity_intl','bond_lt','bond_lt','bond_it','bond_it','equity','equity_intl','bond_lt','equity','equity','equity_intl','equity','gold','bond_lt','bond_lt', 'equity', 'bond_lt', 'commodity']
st.session_state.weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# my strats
# st.session_state.shares=   ['SPY','QQQ','MDY','EFA','FXI','FEMKX','VNQ','GLD','TLT','LQD','VBMFX', '', '', '', '']
# st.session_state.shareclass=['equity','equity','equity','equity_intl','equity_intl','equity_intl','equity','gold','bond_lt','bond_lt','bond_lt','','','','']
# st.session_state.weights=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
st.session_state.benchmark=''
st.session_state.riskparity=False
st.session_state.riskparity_nested=False
st.session_state.rotationstrat=False
st.session_state.uniform=False
st.session_state.vanillariskparity=False
st.session_state.onlystocks=False
st.session_state.sixtyforty=False
st.session_state.trend_u=False
st.session_state.absmom_u=False
st.session_state.relmom_u=False
st.session_state.momtrend_u=False
st.session_state.trend_rp=False
st.session_state.absmom_rp=False
st.session_state.relmom_rp=False
st.session_state.momtrend_rp=False
st.session_state.GEM=False
st.session_state.acc_dualmom=False
st.session_state.acc_dualmom2=False
st.session_state.rot_adm=False
st.session_state.rot_adm_dual_rp=False
st.session_state.TESTING=True
st.session_state.vigilant=False
st.session_state.specific_vigilant=False
st.session_state.specific_rot=False
st.session_state.specific_rot2=False
st.session_state.specific_adm=False
st.session_state.specific_adm_grad_div=False
st.session_state.specific_fabio_adm2=False
st.session_state.specific_fabiofg_adm2=False
st.session_state.specific_fabio_adm3=False
st.session_state.specific=False
# report
st.session_state.create_report=True
st.session_state.report_name='backtest report'
st.session_state.user='FG'
st.session_state.memo='backtest report'
# advanced parameters
st.session_state.MARGINLOAN_INT_RATE=1.5
st.session_state.DAYS_IN_YEAR=252
st.session_state.DAYS_IN_YEAR_BOND_PRICE=360
st.session_state.reb_period_days=1
st.session_state.reb_period_years=1
st.session_state.reb_period_custweights=1
st.session_state.lookback_period_short_days=20
st.session_state.lookback_period_short_years=5
st.session_state.lookback_period_short_custweights=20
st.session_state.lookback_period_long_days=120
st.session_state.lookback_period_long_years=10
st.session_state.lookback_period_long_custweights=120
st.session_state.moving_average_period_days=252
st.session_state.moving_average_period_years=5
st.session_state.moving_average_period_custweights=252
st.session_state.momentum_period_days=252
st.session_state.momentum_period_years=5
st.session_state.momentum_period_custweights=252
st.session_state.momentum_percentile_days=0.5
st.session_state.momentum_percentile_years=0.5
st.session_state.momentum_percentile_custweights=0.5
st.session_state.corrmethod_days='pearson'
st.session_state.corrmethod_years='pearson'
st.session_state.corrmethod_custweights='pearson'
st.session_state.riskfree=0.01
st.session_state.targetrate=0.01
st.session_state.alpha=0.05
st.session_state.market_mu=0.07
st.session_state.market_sigma=0.15
st.session_state.stddev_sample=True
st.session_state.annualize=True
st.session_state.logreturns=False
st.session_state.assets_startdate=datetime.strptime('2010-01-01', '%Y-%m-%d')
st.session_state.assets_enddate = datetime.strptime('2021-01-01', '%Y-%m-%d')
st.session_state.assets_multiselect=['SP500','ZB.F','ZN.F','BM.F','GC.C']
st.session_state.userallocation_live=False
st.session_state.ADM_live=False

def app():
    global input_df
    global account

    st.title('Home')

    st.write('First adjust the backtest parameters on the left, then launch the backtest by pressing the button below.')

    st.header("Backtest parameters")

    with st.form("input_params"):

        st.session_state.live_backtest = st.checkbox("Backtest live performance from IBKR", value=st.session_state.live_backtest, key='live_backtest_box', help='Perform a backtesting of the performance of live portfolio from IBKR against a strategy ')
        st.markdown("**Choose backtesting period**")
        col1, col2 = st.columns(2)
        st.session_state.startdate = col1.date_input('start date', value=st.session_state.startdate, min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'), max_value=date.today(), key='startdate_box', help='start date of the backtest')
        st.session_state.enddate = col2.date_input('end date', value=st.session_state.enddate, min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'), max_value=date.today(), key='enddate_box', help='end date of the backtest')

        st.markdown("**Choose initial cash and regular contributions or withdrawals**")
        col1, col2 = st.columns(2)
        st.session_state.initial_cash = col1.number_input("initial cash", min_value=0.0, max_value=None, value=st.session_state.initial_cash, step=1000.0, format='%f', key='initial_cash_box', help='initial cash')
        st.session_state.contribution = col2.number_input("contribution or withdrawal", min_value=None, max_value=None, value=st.session_state.contribution, format='%f',step=0.01, key='contribution_box', help='contribution or withdrawal. Can be specified as % of the portfolio value or in absolute terms.')

        st.markdown("**Set a leverage and an expense ratio**")
        col1, col2 = st.columns(2)
        st.session_state.leverage = col1.number_input("leverage", min_value=0.0, max_value=None,step=0.01, value=st.session_state.leverage, format='%f', key='leverage_box', help='daily leverage to apply to assets returns')
        st.session_state.expense_ratio = col2.number_input("expense ratio", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.expense_ratio, format='%f', key='expense_ratio_box', help='annual expense ratio')

        st.markdown("**Choose the data source for the prices**")
        if st.session_state.historic == "Yahoo Finance (daily prices)":
            idx = 0
        elif st.session_state.historic == "Historical DB (daily prices)":
            idx = 1
        else:
            idx = 2

        st.session_state.historic = st.radio('data source', ( "Yahoo Finance (daily prices)", "Historical DB (daily prices)", "Historical DB (yearly prices)"), index=idx, key='historic_box', help='choose the data source')
        if st.session_state.historic == "Yahoo Finance (daily prices)":
            historic_cd = None
        elif st.session_state.historic == "Historical DB (daily prices)":
            historic_cd = 'medium'
        elif st.session_state.historic == "Historical DB (yearly prices)":
            historic_cd = 'long'

        st.markdown("**Choose the assets to include in the portfolio**")

        col1, col2, col3 = st.columns(3)
        col1.write("ticker")
        col2.write("share class")
        col3.write("weight")

        shareclass_options = ('','equity', 'equity_intl', 'bond_it', 'bond_lt', 'gold', 'commodity', 'money_market')
        for i in range(0, len(st.session_state.shares)):
            st.session_state.shares[i] = col1.text_input('',value=st.session_state.shares[i], max_chars=None, key="shares_"+str(i)+"_box")
            st.session_state.shareclass[i] = col2.selectbox('',options=shareclass_options, index=shareclass_options.index(st.session_state.shareclass[i]), key="shareclass_"+str(i)+"_box")
            st.session_state.weights[i] = col3.number_input('',value=st.session_state.weights[i], min_value=0.0, max_value=1.0, step=0.01, format='%f', key="weights_"+str(i)+"_box")

        st.markdown('- **ticker** is the ticker from Yahoo Finance (if this is the selected price source)')
        st.markdown('- **share class** is the asset type. Possibilities are `equity, equity_intl, bond_lt, bond_it, gold, commodity`, where "bond_lt" and "bond_it" are long and intermediate duration bonds, respectively. __This argument is mandatory when the data source is Yahoo Finance.')
        st.markdown('- **weight** is the portfolio weight for specified asset (e.g. `0.35`). The weights need to sum to 1. When weights are specified a custom weights strategy is used that simply loads the weights specified. Alternative is to use a pre-defined strategy.')

        st.session_state.benchmark = st.text_input("benchmark", value=st.session_state.benchmark, max_chars=None, key='benchmark_box', help='ticker of a benchmark')

        st.subheader("Strategies")

        st.session_state.riskparity = st.checkbox('risk parity', value=st.session_state.riskparity, key='riskparity_box', help='Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run at portfolio level.')
        st.session_state.riskparity_nested = st.checkbox('risk parity nested', value=st.session_state.riskparity_nested, key='riskparity_nested_box', help='Dynamic allocation of weights according to the risk parity methodology (see https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/). Here the risk parity is run first at asset classe level (for assets belonging to the same asset class) and then at portfolio level.')
        st.session_state.rotationstrat = st.checkbox('asset rotation', value=st.session_state.rotationstrat, key='rotationstrat_box', help='Asset rotation strategy that buy either gold, bonds or equities based on a signal (see https://seekingalpha.com/article/4283733-simple-rules-based-asset-rotation-strategy). To use this strategy tick the box signal assets.')
        st.session_state.uniform = st.checkbox('uniform', value=st.session_state.uniform, key='uniform_box', help='Static allocation uniform across asset classes. Assets are allocated uniformly within the same asset class.')
        st.session_state.vanillariskparity = st.checkbox('static risk parity', value=st.session_state.vanillariskparity, key='vanillariskparity_box', help='Static allocation to asset classes where weights are taken from https://www.theoptimizingblog.com/leveraged-all-weather-portfolio/ (see section "True Risk Parity").')
        st.session_state.onlystocks = st.checkbox('only equity', value=st.session_state.onlystocks, key='onlystocks_box', help='Static allocation only to the equity class. Assets are allocated uniformly within the equity class.')
        st.session_state.sixtyforty = st.checkbox('60% equities 40% bonds', value=st.session_state.sixtyforty, key='sixtyforty_box', help='Static allocation 60% to the equity class, 20% to the Long Term Bonds class and 20% to the Short Term Bonds class. Assets are allocated uniformly within the asset classes.')
        st.session_state.trend_u = st.checkbox('trend uniform', value=st.session_state.trend_u, key='trend_u_box', help='First weights are assigned according to the "uniform" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        st.session_state.absmom_u = st.checkbox('absolute momentum uniform', value=st.session_state.absmom_u, key='absmom_u_box', help='First weights are assigned according to the "uniform" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).')
        st.session_state.relmom_u = st.checkbox('relative momentum uniform', value=st.session_state.relmom_u, key='relmom_u_box', help='First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "uniform" strategy.')
        st.session_state.momtrend_u = st.checkbox('relative momentum & trend uniform', value=st.session_state.momtrend_u, key='momtrend_u_box', help='First weights are assigned according to the "uniform" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        st.session_state.trend_rp = st.checkbox('trend risk parity', value=st.session_state.trend_rp, key='trend_rp_box', help='First weights are assigned according to the "riskparity" strategy. Then, if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        st.session_state.absmom_rp = st.checkbox('absolute momentum risk parity', value=st.session_state.absmom_rp, key='absmom_rp_box', help='First weights are assigned according to the "riskparity" strategy. Then, if the asset return over the period (momentum) is less than 0, the weight is set to zero (leave as cash).')
        st.session_state.relmom_rp = st.checkbox('relative momentum risk parity', value=st.session_state.relmom_rp, key='relmom_rp_box', help='First assets are ranked based on their return over the period (momentum) and divided in two classes. The portfolio is formed by the assets belonging to the higher return class. Then, weights are assigned to this portfolio according to the "risk parity" strategy.')
        st.session_state.momtrend_rp = st.checkbox('relative momentum & trend risk parity', value=st.session_state.momtrend_rp, key='momtrend_rp_box', help='First weights are assigned according to the "riskparity" strategy. Second, assets are ranked based on their return over the period (momentum) and divided in two classes. For the assets belonging to the lower return class, the weight is set to zero (leave as cash). Finally, a trend filter is then applied to assets with positive weight: if the current asset price is smaller than the simple moving average, the weight is set to zero (leave as cash).')
        st.session_state.GEM = st.checkbox('Global equity momentum', value=st.session_state.GEM, key='GEM_box', help='Global equity momentum strategy. Needs only 4 assets of classes equity, equity_intl, bond_lt, money_market. example: `VEU,IVV,BIL,AGG equity_intl,equity,money_market,bond_lt`. See https://blog.thinknewfound.com/2019/01/fragility-case-study-dual-momentum-gem/')
        st.session_state.acc_dualmom = st.checkbox('Accelerating Dual Momentum', value=st.session_state.acc_dualmom, key='acc_dualmom_box', help='Accelerating Dual Momentum. Needs only 3 assets of classes equity, equity_intl, bond_lt. example: VFINX,VINEX,VUSTX, shareclass equity,equity_intl,bond_lt. See https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/')
        st.session_state.acc_dualmom2 = st.checkbox('Accelerating Dual Momentum (extended)', value=st.session_state.acc_dualmom2, key='acc_dualmom2_box', help='Accelerating Dual Momentum (extended). Needs only 4 assets of classes equity, equity_intl, bond_lt, gold. example: VFINX,VINEX,VUSTX,GLD shareclass equity,equity_intl,bond_lt,gold.')
        st.session_state.rot_adm = st.checkbox('Rotation + Accelerating Dual Momentum', value=st.session_state.rot_adm, key='rot_adm_box', help='Rotation Strategy + Accelerating Dual Momentum. If the strategies do not agree, assign 50% to each. Needs only 4 assets of classes equity, equity_intl, bond_lt, gold. example: VFINX,VINEX,VUSTX,GLD shareclass equity,equity_intl,bond_lt,gold.')
        st.session_state.rot_adm_dual_rp = st.checkbox('Rotation + Accelerating Dual Momentum + Antonacci dual momentum + risk parity', value=st.session_state.rot_adm_dual_rp, key='rot_adm_dual_rp_box', help='Rotation Strategy + Accelerating Dual Momentum  + Antonacci dual momentum + risk parity. It assigns 25% weight to each. Needs only 4 assets of classes equity, equity_intl, bond_lt, gold, commodity, money_market. example: VFINX,VINEX,VUSTX,GLD,GSG,BIL shareclass equity,equity_intl,bond_lt,gold, commodity, money_market.')
        st.session_state.vigilant = st.checkbox('Vigilant asset allocation', value=st.session_state.vigilant, key='vigilant_box', help='Vigilant asset allocation from https://allocatesmartly.com/vigilant-asset-allocation-dr-wouter-keller-jw-keuning/. Example: SPY,EFA,EEM,AGG,LQD,IEF,SHY shareclass equity,equity_intl,equity_intl,bond_lt,bond_lt,bond_it,bond_it.')
        st.session_state.TESTING = st.checkbox('TESTING', value=st.session_state.TESTING, key='TESTING_box', help='allocation used for testing purposes')
        st.session_state.specific_vigilant = st.checkbox('vigilant (specific assets)', value=st.session_state.specific_vigilant, key='specific_vigilant_box', help='vigilant asset allocation with specific assets')
        st.session_state.specific_rot = st.checkbox('Rotation Strategy (specific assets)', value=st.session_state.specific_rot, key='specific_rot_box', help='Rotation Strategy with specific assets')
        st.session_state.specific_rot2 = st.checkbox('Rotation Strategy diversified (specific assets)', value=st.session_state.specific_rot2, key='specific_rot2_box', help='Rotation Strategy diversified with specific assets')
        st.session_state.specific_adm = st.checkbox('ADM (specific assets)', value=st.session_state.specific_adm, key='specific_adm_box', help='ADM with specific assets')
        st.session_state.specific_adm_grad_div = st.checkbox('Accelerating Dual Momentum gradual & diversified (specific assets)', value=st.session_state.specific_adm_grad_div, key='specific_adm_grad_div_box', help='Accelerating Dual Momentum gradual & diversified with specific assets')
        st.session_state.specific_fabio_adm2 = st.checkbox('ADM Fabio 2 (specific assets)', value=st.session_state.specific_fabio_adm2, key='specific_fabio_adm2_box', help='Accelerating Dual Momentum Fabio 2 with specific assets')
        st.session_state.specific_fabiofg_adm2 = st.checkbox('ADM Fabio 2 diversified (specific assets)', value=st.session_state.specific_fabiofg_adm2, key='specific_fabiofg_adm2_box', help='Accelerating Dual Momentum Fabio 2 diversified with specific assets')
        st.session_state.specific_fabio_adm3 = st.checkbox('ADM Fabio 3 (specific assets)', value=st.session_state.specific_fabio_adm3, key='specific_fabio_adm3_box', help='ADM Fabio 3 with specific assets')
        st.session_state.specific = st.checkbox('hold specific assets', value=st.session_state.specific, key='specific_box', help='simple specific assets strategy, holding the assets')

        st.subheader("HTML Report")
        # st.session_state.create_report = st.checkbox('create PDF report', value=st.session_state.create_report, key='create_report', help=None)
        st.session_state.report_name = st.text_input("report name", value=st.session_state.report_name, max_chars=None, key='report_name_box', help=None)
        st.session_state.user = st.text_input("user name", value=st.session_state.user, max_chars=None, key='user_box', help='user generating the report')
        st.session_state.memo = st.text_input("report memo", value=st.session_state.memo, max_chars=None, key='memo_box', help='description of the report')

        #launch_btn = st.button("Launch backtest")
        launch_btn = st.form_submit_button("Launch backtest")

    if launch_btn:
        connected_flg = False
        if st.session_state.ib_client is not None:
            connected_flg = True
            # if st.session_state.ib_client.is_authenticated()['authenticated']:
            #     connected_flg = True

        if st.session_state.live_backtest and not connected_flg:
            st.write("Please connect to Interactive Brokers to perform a backtest of the real trading portfolio.")
            st.stop()
        else:
            params['live_backtest'] = st.session_state.live_backtest
            params['ib_client'] = st.session_state.ib_client
            params['startdate'] = st.session_state.startdate
            params['enddate'] = st.session_state.enddate
            params['initial_cash'] = st.session_state.initial_cash
            params['contribution'] = st.session_state.contribution
            params['leverage'] = st.session_state.leverage
            params['expense_ratio'] = st.session_state.expense_ratio
            params['historic'] = historic_cd
            params['shares'] = st.session_state.shares
            params['shareclass'] = st.session_state.shareclass
            params['weights'] = st.session_state.weights
            params['benchmark'] = st.session_state.benchmark
            params['riskparity'] = st.session_state.riskparity
            params['riskparity_nested'] = st.session_state.riskparity_nested
            params['rotationstrat'] = st.session_state.rotationstrat
            params['uniform'] = st.session_state.uniform
            params['vanillariskparity'] = st.session_state.vanillariskparity
            params['onlystocks'] = st.session_state.onlystocks
            params['sixtyforty'] = st.session_state.sixtyforty
            params['trend_u'] = st.session_state.trend_u
            params['absmom_u'] = st.session_state.absmom_u
            params['relmom_u'] = st.session_state.relmom_u
            params['momtrend_u'] = st.session_state.momtrend_u
            params['trend_rp'] = st.session_state.trend_rp
            params['absmom_rp'] = st.session_state.absmom_rp
            params['relmom_rp'] = st.session_state.relmom_rp
            params['momtrend_rp'] = st.session_state.momtrend_rp
            params['GEM'] = st.session_state.GEM
            params['acc_dualmom'] = st.session_state.acc_dualmom
            params['acc_dualmom2'] = st.session_state.acc_dualmom2
            params['rot_adm'] = st.session_state.rot_adm
            params['rot_adm_dual_rp'] = st.session_state.rot_adm_dual_rp
            params['vigilant'] = st.session_state.vigilant
            params['TESTING'] = st.session_state.TESTING
            params['specific_vigilant'] = st.session_state.specific_vigilant
            params['specific_rot'] = st.session_state.specific_rot
            params['specific_rot2'] = st.session_state.specific_rot2
            params['specific_adm'] = st.session_state.specific_adm
            params['specific_adm_grad_div'] = st.session_state.specific_adm_grad_div
            params['specific_fabio_adm2'] = st.session_state.specific_fabio_adm2
            params['specific_fabiofg_adm2'] = st.session_state.specific_fabiofg_adm2
            params['specific_fabio_adm3'] = st.session_state.specific_fabio_adm3
            params['specific'] = st.session_state.specific
            # reports params
            params['create_report'] = st.session_state.create_report
            params['report_name'] = st.session_state.report_name
            params['user'] = st.session_state.user
            params['memo'] = st.session_state.memo
            # advanced params
            params['interest_rate'] = st.session_state.MARGINLOAN_INT_RATE
            params['DAYS_IN_YEAR'] = st.session_state.DAYS_IN_YEAR
            params['DAYS_IN_YEAR_BOND_PRICE'] = st.session_state.DAYS_IN_YEAR_BOND_PRICE
            params['reb_period_days'] = st.session_state.reb_period_days
            params['reb_period_years'] = st.session_state.reb_period_years
            params['reb_period_custweights'] = st.session_state.reb_period_custweights
            params['lookback_period_short_days'] = st.session_state.lookback_period_short_days
            params['lookback_period_short_years'] = st.session_state.lookback_period_short_years
            params['lookback_period_short_custweights'] = st.session_state.lookback_period_short_custweights
            params['lookback_period_long_days'] = st.session_state.lookback_period_long_days
            params['lookback_period_long_years'] = st.session_state.lookback_period_long_years
            params['lookback_period_long_custweights'] = st.session_state.lookback_period_long_custweights
            params['moving_average_period_days'] = st.session_state.moving_average_period_days
            params['moving_average_period_years'] = st.session_state.moving_average_period_years
            params['moving_average_period_custweights'] = st.session_state.moving_average_period_custweights
            params['momentum_period_days'] = st.session_state.momentum_period_days
            params['momentum_period_years'] = st.session_state.momentum_period_years
            params['momentum_period_custweights'] = st.session_state.momentum_period_custweights
            params['momentum_percentile_days'] = st.session_state.momentum_percentile_days
            params['momentum_percentile_years'] = st.session_state.momentum_percentile_years
            params['momentum_percentile_custweights'] = st.session_state.momentum_percentile_custweights
            params['corrmethod_days'] = st.session_state.corrmethod_days
            params['corrmethod_years'] = st.session_state.corrmethod_years
            params['corrmethod_custweights'] = st.session_state.corrmethod_custweights
            params['riskfree'] = st.session_state.riskfree
            params['targetrate'] = st.session_state.targetrate
            params['alpha'] = st.session_state.alpha
            params['market_mu'] = st.session_state.market_mu
            params['market_sigma'] = st.session_state.market_sigma
            params['stddev_sample'] = st.session_state.stddev_sample
            params['annualize'] = st.session_state.annualize
            params['logreturns'] = st.session_state.logreturns

        #if input_df != 0:
    mainout = main(params)
    if mainout is not False:
        input_df = copy.deepcopy(mainout)

        if params['live_backtest']:
            live_returns = utils.get_portfolio_returns(st.session_state.ib_client, st.session_state.ib_client.account, freq='D')
            live_returns = live_returns[live_returns['date'] > params['startdate']]
            live_nav = utils.get_portfolio_nav(st.session_state.ib_client, st.session_state.ib_client.account, freq='D')
            live_nav = live_nav[live_nav['date'] > params['startdate']]
            fig = px.line(live_nav, x="date", y="nav")

            st.markdown("### Real portfolio NAV (CHF)")
            st.plotly_chart(fig, use_container_width=True)

        # Portfolio value
        idx = 0

        if params['live_backtest']:
            live_price = params['initial_cash'] * (1 + live_returns['returns']).cumprod()
            live_price.iloc[0] = params['initial_cash']
            live_price_df = pd.DataFrame({'date':live_returns['date'],'real_portfolio':live_price})
            input_df[idx] = pd.merge(input_df[idx], live_price_df, left_index=True, right_on='date', how="inner")
            input_df[idx].set_index(['date'], inplace=True)

        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index

        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='strategy', value_name='price')

        fig = px.line(input_df_long, x="date", y="price", color="strategy")

        st.markdown("### Portfolio value")
        st.plotly_chart(fig, use_container_width=True)

        # Portfolio drawdowns
        idx = 5 # find a smarter way later

        if params['live_backtest']:
            live_portfolio_dd = (live_price.shift(-1) / live_price.cummax(axis=0) - 1).shift(1)
            live_portfolio_dd.iloc[0] = 0
            live_portfolio_dd = live_portfolio_dd.clip(None, 0)
            live_portfolio_dd_df = pd.DataFrame({'date': live_returns['date'], 'real_portfolio': live_portfolio_dd})
            input_df[idx] = pd.merge(input_df[idx], live_portfolio_dd_df, left_index=True, right_on='date', how="inner")
            input_df[idx].set_index(['date'], inplace=True)

        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index

        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='strategy', value_name='drawdown')

        fig = px.line(input_df_long, x="date", y="drawdown", color="strategy")

        st.markdown("### Portfolio drawdown")
        st.plotly_chart(fig, use_container_width=True)

        # Portfolio metrics
        st.markdown("### Portfolio metrics")

        if params['live_backtest']:
            returns_df = live_returns['returns']
            live_portfolio_metrics_df = utils.live_portfolio_metrics(params, returns_df)

            input_df[2] = pd.merge(input_df[2], live_portfolio_metrics_df, left_index=True, right_index=True, how = 'inner')

        st.dataframe(input_df[2])


        # Portfolio weights
        st.markdown("### Portfolio weights")
        # col1, col2 = st.columns(2)
        #
        # idx = 3
        # columns=input_df[idx].columns
        # input_df[idx]['date'] = input_df[idx].index
        # input_df_long = pd.melt(input_df[idx], id_vars=['date','strategy'], value_vars=columns[0:-1],var_name='asset', value_name='weight')
        #
        # col1.subheader("Target weights")
        #
        # for strat in input_df_long['strategy'].unique():
        #     fig = px.bar(input_df_long[input_df_long['strategy']==strat], x="date", y="weight", color="asset", title=strat + ' weights')
        #     col1.plotly_chart(fig, use_container_width=True)
        idx = 4
        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date','strategy'], value_vars=columns[0:-1],var_name='asset', value_name='weight')

        st.subheader("Effective weights")

        for strat in input_df_long['strategy'].unique():
            fig = px.bar(input_df_long[input_df_long['strategy']==strat], x="date", y="weight", color="asset", title=strat + ' weights')
            st.plotly_chart(fig, use_container_width=True)


        # Asset value
        idx = 6
        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='asset', value_name='price')

        fig = px.line(input_df_long, x="date", y="price", color="asset")

        st.markdown("### Assets value")
        st.plotly_chart(fig, use_container_width=True)

        # Assets drawdowns
        idx = 7 # find a smarter way later
        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='asset', value_name='drawdown')

        fig = px.line(input_df_long, x="date", y="drawdown", color="asset")

        st.markdown("### Assets drawdown")
        st.plotly_chart(fig, use_container_width=True)

        # # Portfolio Returns
        idx = 1
        # Determine the price frequency
        dates=[]
        for i in range(1, len(input_df[idx].index)):
            dates.append(datetime.strptime(str(input_df[idx].index[i]), '%Y-%m-%d'))
        datediff = stats.mode(np.diff(dates))[0][0]
        if datediff > timedelta(days=250):
            frequency = "Years"
        elif datediff < timedelta(days=2):
            frequency = "Days"

        rolling_ret_period = st.slider("rolling returns period (in years)", min_value=1, max_value=30,
                                       value=1, step=1, format='%i', key='rolling_ret_period',
                                       help='period of rolling annual return (in years)')

        if frequency == "Days": # plot the rolling return (annualized)
            for column in input_df[idx]:
                if params['logreturns']:
                    input_df[idx][column] = (input_df[idx][column]).rolling(window=params['DAYS_IN_YEAR']*rolling_ret_period).sum()/rolling_ret_period
                else:
                    input_df[idx][column] = (1 + input_df[idx][column]).rolling(window=params['DAYS_IN_YEAR']*rolling_ret_period).apply(np.prod) ** (1 / rolling_ret_period) - 1
        elif frequency == "Years": # plot the rolling 5 years return
            for column in input_df[idx]:
                if params['logreturns']:
                    input_df[idx][column] = (input_df[idx][column]).rolling(window=rolling_ret_period).mean()
                else:
                    input_df[idx][column] = (1 + input_df[idx][column]).rolling(window=rolling_ret_period).apply(np.prod) - 1


        columns=input_df[idx].columns
        input_df[idx]['date'] = input_df[idx].index
        input_df_long = pd.melt(input_df[idx], id_vars=['date'], value_vars=columns,var_name='strategy', value_name='rolling return')

        fig = px.line(input_df_long, x="date", y="rolling return", color="strategy")

        st.markdown("### Portfolio returns")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Downloads area")

        today_str = datetime.today().strftime('%Y-%m-%d')
        outputfilename = ["Fund Prices", "Returns", "Performance Metrics", "Target Weights",
                          "Effective Weights", "Portfolio Drawdown", "Asset Prices", "Assets drawdown"]

        i = 0
        for name in outputfilename:
            inputfilepath = name + "_" + today_str + '.csv'
            tmp_download_link = utils.download_link(input_df[i], inputfilepath, 'Click here to download ' + name)
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            i = i + 1

        inputfilepath = params['report_name'] + "_" + today_str + '.html'
        tmp_download_link = utils.download_link(input_df[8], inputfilepath, 'Click here to download the html report')
        st.markdown(tmp_download_link, unsafe_allow_html=True)