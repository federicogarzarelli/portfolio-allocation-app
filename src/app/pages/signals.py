import streamlit as st
from scipy import stats
import os, sys
import plotly.express as px
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image
from pages.home import session_state

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import SessionState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils

def app():
    st.title('Market Signals')

    st.write('Useful market signals from popular strategies.')

    st.markdown("## Rotational strategy")
    st.write("Signal for the asset rotation strategy that buys either gold, bonds or equities"
             " (see https://seekingalpha.com/article/4283733-simple-rules-based-asset-rotation-strategy).")

    rotstrat_curves = utils.load_economic_curves(date.today()-timedelta(180), date.today())
    rotstrat_curves = rotstrat_curves.drop(['Max'],axis=1)
    columns = rotstrat_curves.columns
    rotstrat_curves['date'] = rotstrat_curves.index
    rotstrat_curves_long = pd.melt(rotstrat_curves, id_vars=['date'], value_vars=columns, var_name='asset', value_name='signal')

    fig = px.line(rotstrat_curves_long, x="date", y="signal", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.beta_columns(2)

    col1.markdown("- If the yield curve (T10Y2Y) is highest, buy stocks. ")
    col1.markdown("- If the inflation expectation rate minus the yield curve (T10YIE_T10Y2Y) is highest, buy gold.")
    col1.markdown("- If the 20-year TIP rate (DFII20) is highest, buy long-term bonds.")
    RotStrat_stocks = [['DFII20', '20-Year Treasury Inflation-Indexed Security, Constant Maturity','https://fred.stlouisfed.org/series/DFII20'],
                      ['T10Y2Y', '10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity','https://fred.stlouisfed.org/series/T10Y2Y'],
                  ['T10YIE_T10Y2Y', '10-Year Breakeven Inflation Rate minus T10Y2Y','https://fred.stlouisfed.org/series/T10YIE']]
    RotStrat_stocks_df = pd.DataFrame(RotStrat_stocks, columns=['ticker', 'name', 'link'])
    col2.dataframe(RotStrat_stocks_df)


    st.markdown("## GEM")
    st.write("Global equity momentum strategy. Needs only 4 assets of classes equity, equity_intl, bond_lt, money_market. "
             "example: `VEU,IVV,BIL,AGG equity_intl,equity,money_market,bond_lt`. "
             "See https://blog.thinknewfound.com/2019/01/fragility-case-study-dual-momentum-gem/")

    GEM_curves = utils.load_GEM_curves(date.today()-timedelta(365*2), date.today())
    fig = px.line(GEM_curves, x="date", y="return", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.beta_columns(2)
    wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    GEM_image_path = utils.find('GEM.png', wd)
    GEM_image = Image.open(GEM_image_path)
    col1.image(GEM_image)
    GEM_stocks = [['VEU', 'Vanguard FTSE All-World ex-US Index Fund ETF Shares ','https://finance.yahoo.com/quote/VEU'],
                  ['IVV', 'iShares Core S&P 500 ETF','https://finance.yahoo.com/quote/IVV'],
                  ['BIL', 'SPDR Bloomberg Barclays 1-3 Month T-Bill ETF','https://finance.yahoo.com/quote/BIL']]
    GEM_stocks_df = pd.DataFrame(GEM_stocks, columns=['ticker', 'name', 'link'])
    col2.dataframe(GEM_stocks_df)

    st.markdown("## Accelerating dual momentum")
    st.write("Accelerating Dual Momentum. Needs only 3 assets of classes equity, equity_intl, bond_lt. example: "
             "VFINX,VINEX,VUSTX, shareclass equity,equity_intl,bond_lt. "
             "See https://engineeredportfolio.com/2018/05/02/accelerating-dual-momentum-investing/")

    AccDualMom_curves = utils.load_AccDualMom_curves(date.today()-timedelta(365*2), date.today())
    fig = px.line(AccDualMom_curves, x="date", y="score", color="asset")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.beta_columns(2)
    wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    AccDualMom_image_path = utils.find('acc_dualmom.png', wd)
    AccDualMom_image = Image.open(AccDualMom_image_path)
    col1.image(AccDualMom_image)
    AccDualMom_stocks = [['VFINX', 'Vanguard 500 Index Fund Investor Shares','https://finance.yahoo.com/quote/VFINX'],
                  ['VINEX', 'Vanguard International Explorer Fund Investor Shares','https://finance.yahoo.com/quote/VINEX'],
                  ['VUSTX', 'Vanguard Long-Term Treasury Fund Investor Shares','https://finance.yahoo.com/quote/VUSTX']]
    AccDualMom_stocks_df = pd.DataFrame(AccDualMom_stocks, columns=['ticker', 'name', 'link'])
    col2.dataframe(AccDualMom_stocks_df)




