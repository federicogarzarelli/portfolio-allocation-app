import streamlit as st
import os, sys
# from pages.home import session_state
import pages.home
from datetime import date, datetime
from GLOBAL_VARS import *
from PortfolioDB import PortfolioDB
import numpy as np
import plotly.express as px
import utils
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
# import SessionState

def app():
    st.title('Explore Prices DB')

    st.write('Here you can explore the database of prices.')

    st.markdown('## Assets info')

    db = PortfolioDB(databaseName=DB_NAME)
    sqlQry = "SELECT A.ticker, A.name, A.asset_class, B.min_dt, B.max_dt \
                FROM DIM_STOCKS as A INNER JOIN DIM_STOCK_DATES as B \
                ON A.ticker = B.ticker"
    data = db.readDatabase(sqlQry)
    st.dataframe(data)

    st.markdown('## Assets metrics')

    with st.form("assets_input_params"):

        col1, col2 = st.columns(2)
        st.session_state.assets_startdate = col1.date_input('start date', value=st.session_state.assets_startdate,
                                                        min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'),
                                                        max_value=date.today(), key='assets_startdate_box',
                                                        help='start date for the asset chart')
        st.session_state.assets_enddate = col2.date_input('end date', value=st.session_state.assets_enddate,
                                                      min_value=datetime.strptime('1900-01-01', '%Y-%m-%d'),
                                                      max_value=date.today(), key='assets_enddate_box',
                                                      help='end date for the asset chart')

        st.session_state.assets_multiselect = st.multiselect("Select the assets", options=data['ticker'],
                                                          default=None, key='assets_multiselect_box',
                                                          help="Select the assets to display")

        launch_assets_btn = st.form_submit_button("check assets")

    if launch_assets_btn:
        assets = st.session_state.assets_multiselect
        assets_str = '","'.join([str(elem) for elem in assets])
        assets_str = '"' + assets_str + '"'
        sqlQry="SELECT A.date, A.ticker, A.close as price, B.frequency, B.name, B.asset_class, B.treatment_type FROM FACT_HISTPRICES AS A" \
               " INNER JOIN DIM_STOCKS AS B ON A.ticker = B.ticker WHERE A.ticker IN (" + \
               assets_str + ") and A.date BETWEEN '" + st.session_state.assets_startdate.strftime("%Y-%m-%d") + \
               "' and '" + st.session_state.assets_enddate.strftime("%Y-%m-%d") + "'"

        # Plot the price
        asset_data = db.readDatabase(sqlQry)

        fig = px.line(asset_data, x="date", y="price", color="ticker")

        st.markdown("### Assets value")
        st.plotly_chart(fig, use_container_width=True)

        # Plot the returns
        grouping_flds = ['ticker','name','asset_class','treatment_type','frequency']
        asset_data['returns'] = asset_data.sort_values('date').groupby(grouping_flds).price.pct_change()
        asset_data['returns_1'] = asset_data['returns'] + 1
        asset_data['factor'] = np.where(asset_data['frequency'] == 'D', params['DAYS_IN_YEAR'], 1)

        # Create table of metrics
        # cagr
        cagr = asset_data.sort_values('date').groupby(['ticker']).returns_1.apply(np.prod) ** (asset_data.sort_values('date').groupby(['ticker']).factor.first() / asset_data.sort_values('date').groupby(['ticker']).returns_1.count()) - 1

        # volatility
        volatility = asset_data.sort_values('date').groupby(['ticker']).returns.std()*np.sqrt(asset_data.sort_values('date').groupby(['ticker']).factor.first())

        sr = (cagr - 0.01) / volatility

        # cagr = cagr.astype(float).map(lambda n: '{:.2%}'.format(n))
        # volatility = volatility.astype(float).map(lambda n: '{:.2%}'.format(n))
        # sr = sr.astype(float).map(lambda n: '{:.2%}'.format(n))

        # max_dd
        #max_dd = asset_data.sort_values('date').groupby(['ticker','frequency']).returns.apply(utils.max_dd)
        metrics = pd.concat([cagr, volatility, sr], axis=1)
        metrics.columns = ['CAGR', 'Volatility','Sharpe Ratio']

        metrics = (100. * metrics).round(1).astype(str) + '%'
        asset_data_grouped = asset_data.groupby(['ticker']).first()[['name', 'frequency', 'asset_class', 'treatment_type']]
        metrics_joined = pd.merge(left=asset_data_grouped,right=metrics,left_index=True,right_index=True)

        st.markdown("### Assets metrics")

        st.dataframe(metrics_joined)
        st.write("Sharpe ratio calculated as (CAGR - 1%)/Volatility where 1% is assumed to be the risk free rate.")






