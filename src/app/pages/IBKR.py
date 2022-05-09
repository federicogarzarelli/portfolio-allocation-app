import pathlib
from IPython.display import display, HTML
import sys
from pprint import pprint
from configparser import ConfigParser
import math
import pandas as pd
import streamlit as st
import os, sys
import plotly.express as px
import numpy as np

# from pages.home import session_state

pd.options.mode.chained_assignment = None  # default='warn'

MIN_CASH_THRESHOLD = 1000
MIN_FX_CONVERT = 10

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import SessionState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import *
from rebalance import optimal_rebalance_while_buying, standard_rebalance

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from resources.IBRK_client import IBClient

global live_orders
live_orders = ""
# global ib_client
global user_input
#######################################################################################################################
#### Main rebalance function
#######################################################################################################################
def rebalance(positions_df_stk, settled_cash, CHFUSD_quote, account, leverage_live, rebalance_type = 'with_contributions'):
    # Calculate the contribution weights using numerical optimization
    portfolio_value = positions_df_stk['mktValue_USD'].sum()*leverage_live
    contribution = settled_cash['BASE'] * CHFUSD_quote
    weights = positions_df_stk['actual_allocation'].to_numpy()
    target_weights = positions_df_stk['target_allocation'].to_numpy()

    if rebalance_type == 'with_contributions':
        contribution_weights = optimal_rebalance_while_buying(value=portfolio_value, c=contribution, w=weights,
                                                              w_target=target_weights)
    elif rebalance_type == 'standard':
        contribution_weights = standard_rebalance(value=portfolio_value, c=contribution, w=weights,
                                                  w_target=target_weights)

    positions_df_stk['contribution_weights_USD'] = contribution_weights
    positions_df_stk['contribution_weights_CCY'] = positions_df_stk['contribution_weights_USD'] / \
                                                   positions_df_stk['FX_rate']

    # Convert the cash into the currencies required to buy the assets in the portfolio
    CCY_df = pd.DataFrame([[0, 0], [0, 0], [0, 0]],
                          index=['USD', 'EUR', 'CHF'],
                          columns=['AMT', 'AMT_NEEDED'])

    st.markdown("\n"
          "****************************************************\n"
          " CURRENCY CONVERSION                                \n"
          "****************************************************\n"
          )

    for CCY in CCY_df.index:
        if CCY not in settled_cash:
            settled_cash[CCY] = 0
        CCY_df.loc[CCY, 'AMT'] = positions_df_stk.loc[
            positions_df_stk['currency'] == CCY, 'contribution_weights_CCY'].sum()
        CCY_df.loc[CCY, 'AMT_NEEDED'] = CCY_df.loc[CCY, 'AMT'] - settled_cash[CCY]

    for CCY_1 in CCY_df.index:
        for CCY_2 in CCY_df.index:
            if CCY_1 != CCY_2:
                # pprint(CCY_1 + ', ' + CCY_2)
                if CCY_df.loc[CCY_1, 'AMT_NEEDED'] > MIN_FX_CONVERT and CCY_df.loc[CCY_2, 'AMT_NEEDED'] < -MIN_FX_CONVERT:
                    FX = FX_ticker_mapping.loc[(FX_ticker_mapping['ccyfrom'] == CCY_1) & (
                            FX_ticker_mapping['ccyto'] == CCY_2), 'FX_rate'].values[0]
                    amt_ccyfrom = min(CCY_df.loc[CCY_1, 'AMT_NEEDED'] * FX, -CCY_df.loc[CCY_2, 'AMT_NEEDED'])
                    amt_ccyto = amt_ccyfrom / FX
                    order_res = convertFX(session_state.ib_client, ccyfrom=CCY_2, ccyto=CCY_1, amt_ccyfrom=amt_ccyfrom,
                                          account_id=account)
                    order_status = order_res[0][0]['order_status']
                    order_status_management.count = 0
                    order_status_management(session_state.ib_client, order_status, order_details=order_res)
                    if order_status in ["Filled", "Submitted", "PreSubmitted"]:  # assume that the order will be Filled
                        # ignoring transaction costs, which are anyway very small for FX
                        CCY_df.loc[CCY_1, 'AMT_NEEDED'] = CCY_df.loc[CCY_1, 'AMT_NEEDED'] - amt_ccyto
                        CCY_df.loc[CCY_2, 'AMT_NEEDED'] = CCY_df.loc[CCY_2, 'AMT_NEEDED'] + amt_ccyfrom

    # Now we should have all the cash in the right currency to buy stocks
    # Get prices and amounts to buy and ask the user to confirm the order
    st.markdown("\n"
          "****************************************************\n"
          "  GETTING STOCK QUOTES                              \n"
          "****************************************************\n"
          )
    positions_df_stk.set_index('contractDesc',inplace=True)

    stk_conid = []
    stk_quote = []
    stk_amt = []
    stk_side = []
    stk_rth = []

    i = 0
    for stk in positions_df_stk.index:
        this_conid = positions_df_stk.loc[positions_df_stk.index == stk, 'conid'].values[0]
        st.markdown(stk + " contract ID: " + this_conid)
        this_stk_quote = get_stk_quote(session_state.ib_client, conid=this_conid)
        this_minIncrement = get_minIncrement(session_state.ib_client, conid=this_conid)
        this_rth = get_rth(session_state.ib_client, conid=this_conid)
        if contribution_weights[i] > 0:  # If buy round down
            if this_stk_quote > 0:
                this_stk_amt = math.floor(contribution_weights[i] / this_stk_quote)
            else:
                this_stk_amt = 0
            stk_amt.append(this_stk_amt)
            stk_side.append('BUY')
        else:  # if SELL round up
            if this_stk_quote > 0:
                this_stk_amt = math.ceil(contribution_weights[i] / this_stk_quote)
            else:
                this_stk_amt = 0
            stk_amt.append(-this_stk_amt)
            stk_side.append('SELL')
        stk_conid.append(this_conid)
        this_stk_quote = myround(this_stk_quote, prec=2, base=this_minIncrement)
        stk_quote.append(this_stk_quote)
        stk_rth.append(this_rth)
        i += 1

    trades_details = pd.DataFrame(list(zip(stk_conid, stk_quote,
                                           positions_df_stk['currency'], stk_amt, stk_side, stk_rth)),
                                  index=positions_df_stk.index,
                                  columns=['conid', 'quote', 'currency', 'amount', 'side', 'regular_market_hours'])
    # if outside market hours, do not buy or sell stocks.
    trades_details.loc[trades_details['regular_market_hours'] == False, 'amount'] = 0

    st.markdown("\n"
          "See below the details about trades needed for optimally rebalancing my portfolio, using investment "
          "contributions."
          "\n")
    # pprint(trades_details)
    st.dataframe(trades_details)

    st.write("No orders will be sent for stocks outside regular market hours, hence size is set to 0.")

    st.markdown("\n"
                "****************************************************\n"
                "  BUYING and SELLING                                \n"
                "****************************************************\n"
                )
    for stk in trades_details.index:
        if trades_details.loc[stk, 'amount'] > 0:
            st.markdown("\n"
                        " " + stk + "                                              \n"
                                    "****************************************************\n"
                        )
            order_res = stock_buysell(session_state.ib_client, ib_ticker=stk, conid=trades_details.loc[stk, 'conid'],
                                      amt=trades_details.loc[stk, 'amount'], account_id=account,
                                      side=trades_details.loc[stk, 'side'],
                                      price=trades_details.loc[stk, 'quote'])
            st.markdown(trades_details.loc[stk, 'side'] +
                        " order sent for " + stk + ". As soon as the order is filled, a message here and on Telegram will "
                                                   "be sent.")
            order_status = order_res[0][0]['order_status']
            order_status_management.count = 0
            order_status_management(session_state.ib_client, order_status, order_details=order_res)

    return

#######################################################################################################################

# keep this live
@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def check_order_status(ib_client):
    """
    runs every minute and check the status of Submitted, PendingSubmit, PreSubmitted
    :param ib_client:
    :return:
    """
    global live_orders
    threading.Timer(300.0, check_order_status, [ib_client]).start()
    new_live_orders = ib_client.get_live_orders()
    new_live_orders_df = pd.DataFrame.from_records(new_live_orders['orders'])
    if not new_live_orders_df.empty:
        st.markdown("\n"
              "****************************************************\n"
              "  ORDER STATUS CHANGES (runs every 5 minutes)       \n"
              "****************************************************\n"
              )
        new_live_orders_df = new_live_orders_df[['ticker','side','orderType','orderId','filledQuantity','status']]
        live_orders_joined = live_orders.set_index('orderId').join(new_live_orders_df.set_index('orderId'), on='orderId', how='left')
        live_orders_joined['statusChange'] = 0
        live_orders_joined.loc[live_orders_joined['status_old'] != live_orders_joined['status'], 'statusChange'] = 1
        live_orders_joined_status_chg = live_orders_joined.loc[live_orders_joined['statusChange'] == 1, :]
        if not live_orders_joined_status_chg.empty:
            for stk in live_orders_joined_status_chg['ticker']:
                orderType = live_orders_joined_status_chg.loc[live_orders_joined_status_chg['ticker'] == stk, 'orderType']
                side = live_orders_joined_status_chg.loc[live_orders_joined_status_chg['ticker'] == stk, 'side']
                status = live_orders_joined_status_chg.loc[live_orders_joined_status_chg['ticker'] == stk, 'status']
                status_old = live_orders_joined_status_chg.loc[live_orders_joined_status_chg['ticker'] == stk, 'status_old']
                msg = side + " " + orderType + " order on " + stk + " changed status from " + status_old + " to " + status
                log(msg)
                send_telegram(msg)
        else:
            st.markdown("No order status has changed.")
        live_orders = new_live_orders_df[['orderId', 'filledQuantity', 'status']]
        live_orders.columns = ['orderId', 'filledQuantity_old', 'status_old']

def app():
    global live_orders

    # Grab configuration values.
    config = ConfigParser()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    file_path = pathlib.Path(r'config/config.ini').resolve()
    config.read(file_path)

    # Load the details.
    # account = config.get('main', 'PAPER_ACCOUNT')
    # username = config.get('main', 'PAPER_USERNAME')
    account = config.get('main', 'regular_account')
    username = config.get('main', 'regular_username')

    st.title('Execute automated trading in IBKR')

    st.write('Trades automatically based on pre-set strategies or user-defined asset allocation.')

    st.markdown('### Strategies')
    with st.form("input_params"):
        st.write("Choose one of the options below:")
        session_state.userallocation_live=st.checkbox("User defined allocation", value=session_state.userallocation_live, key="userallocation_live", help='Rebalances the portfolio according to a static user defined asset allocation')
        session_state.ADM_live=st.checkbox("Accelerating Dual Momentum", value=session_state.ADM_live, key="ADM_live", help='Rebalances the portfolio according to the Accelerating Dual Momentum strategy.')
        leverage_live = st.number_input("leverage", min_value = 0.0, max_value = None, value = 1.0, step = 0.1, format = '%f', key = 'leverage_live', help = "Leverage to use on the real portfolio.")
        livetrading_btn = st.form_submit_button("Automated trading")

    if livetrading_btn:
        # Check the connection to IBKR
        connected_flg = False
        if session_state.ib_client is not None:
            if session_state.ib_client.is_authenticated()['authenticated']:
                connected_flg = True

        if not connected_flg:
            st.write("Please connect to Interactive Brokers first to lauch the live trading session.")
            connected = False # set the global var to False
            st.stop()
        else:
            ######################################################################################
            # 0. Start the function that periodically checks the order status
            live_orders = session_state.ib_client.get_live_orders()
            live_orders = pd.DataFrame.from_records(live_orders['orders'])
            if live_orders.empty:
                live_orders = pd.DataFrame(columns=['orderId', 'filledQuantity_old', 'status_old'])
            else:
                live_orders = live_orders[['orderId', 'filledQuantity', 'status']]
                live_orders.columns = ['orderId', 'filledQuantity_old', 'status_old']
            check_order_status(session_state.ib_client)

            ######################################################################################
            # 1. Get available cash
            ledger = session_state.ib_client.portfolio_account_ledger(account_id=account)
            ledger_df = pd.DataFrame.from_records(ledger)
            settled_cash = ledger_df.loc['settledcash']

            # Check if there is enough money to buy some stocks
            BUY_FLAG = settled_cash['BASE'] > MIN_CASH_THRESHOLD

            #######################################################################################################################
            # 2. If there is cash to use, calculate the optimal amount to buy to rebalance the portfolio
            if session_state.userallocation_live:
                # Get target allocation and actual positions
                st.markdown("Getting weights for the user defined allocation (in Excel).")
                target_allocation = pd.read_excel('myportfolioallocation.xlsx', index_col=None,
                                                  dtype={'contractDesc': str, 'target_allocation': float})
            elif session_state.ADM_live:
                st.markdown("Getting weights for accelerating dual momentum strategy.")
                target_allocation=AccDualMom_weights()
            else:
                st.markdown("Please specify a strategy to perform automated trading.")

            account_positions = session_state.ib_client.portfolio_account_positions(account_id=account, page_id=0)
            positions_df = pd.DataFrame.from_records(account_positions)

            positions_df=pd.merge(positions_df, target_allocation[['contractDesc','target_allocation']], on='contractDesc', how='outer', indicator=True)
            positions_df.loc[positions_df['_merge']=="right_only",['conid']] = target_allocation.loc[target_allocation['contractDesc'].isin(positions_df.loc[positions_df['_merge']=="right_only",'contractDesc'])]['conid'].tolist()
            positions_df.loc[positions_df['_merge'] == "right_only", ['position']] = 0
            positions_df.loc[positions_df['_merge'] == "right_only", ['mktValue']] = 0
            positions_df.loc[positions_df['_merge'] == "right_only", ['currency']] = 'USD' # dummy currency when market value is 0
            positions_df.loc[positions_df['_merge'] == "right_only", ['assetClass']] = 'STK' # assuming I'm only trading stocks for now

            positions_df['conid'] = positions_df['conid'].astype(int)

            positions_df_stk = positions_df.loc[positions_df['assetClass'] == "STK"]

            # Express everything in USD
            EURUSD_quote = get_FX_quote(session_state.ib_client, ticker='EUR.USD')
            CHFUSD_quote = get_FX_quote(session_state.ib_client, ticker='CHF.USD')
            EURCHF_quote = get_FX_quote(session_state.ib_client, ticker='EUR.CHF')

            FX_ticker_mapping['FX_rate'] = [EURUSD_quote, 1/EURUSD_quote, CHFUSD_quote, 1/CHFUSD_quote, EURCHF_quote, 1/EURCHF_quote]
            positions_df_stk.loc[positions_df_stk['currency'] == 'USD', 'FX_rate'] = 1
            positions_df_stk.loc[positions_df_stk['currency'] == 'EUR', 'FX_rate'] = EURUSD_quote
            positions_df_stk.loc[positions_df_stk['currency'] == 'CHF', 'FX_rate'] = CHFUSD_quote

            positions_df_stk['mktValue_USD'] = positions_df_stk['FX_rate'] * positions_df_stk['mktValue']
            positions_df_stk['actual_allocation'] = positions_df_stk['mktValue_USD'] / positions_df_stk['mktValue_USD'].sum()

            positions_df_stk['target_allocation'] = positions_df_stk['target_allocation'].fillna(0)

            # Plot and display actual vs target allocation
            st.markdown("\n"
                  "****************************************************\n"
                  " CURRENT SITUATION                                  \n"
                  "****************************************************\n"
                  )
            allocation_stk = positions_df_stk[['contractDesc','actual_allocation', 'target_allocation']]
            allocation_stk = allocation_stk.rename(columns={'contractDesc': 'asset'})

            allocation_stk = allocation_stk.melt(id_vars='asset')
            allocation_stk = allocation_stk.rename(columns={'value': 'weight'})

            fig = px.bar(allocation_stk, x="asset", y="weight", color="variable")
            st.plotly_chart(fig, use_container_width=True)

            allocation_stk = positions_df_stk[['contractDesc','actual_allocation', 'target_allocation']]

            allocation_stk['allocation_diff'] = allocation_stk['actual_allocation'] - allocation_stk['target_allocation']
            allocation_stk['allocation_diff_pct'] = allocation_stk['actual_allocation'] / allocation_stk['target_allocation'] - 1

            st.dataframe(allocation_stk)

            # Check if a portfolio rebalance is necessary, following the 5/25 rule by Swedroe
            rule_abs_diff = (allocation_stk['allocation_diff'] > 0.05).any()
            # rule_rel_diff = (allocation_stk['allocation_diff_pct'] > 0.25).any()
            # if rule_abs_diff or rule_rel_diff:
            if rule_abs_diff:
                REBALANCE_FLAG = True
                st.markdown("The porfolio needs rebalancing according to the 5/25 rule by Swedroe.")
            else:
                REBALANCE_FLAG = False
                st.markdown("The porfolio does not needs rebalancing according to the 5/25 rule by Swedroe.")
                st.stop()

            if BUY_FLAG or REBALANCE_FLAG:
                st.markdown("\n"
                      "There are " + str(settled_cash['BASE']) + " CHF cash available for investments (excluding assets to sell). The program will:\n\n"
                      "1. calculate the new allocation to rebalance the portfolio \n"
                      "2. convert available cash into the required currencies \n"
                      "3. buy the assets \n")

                rebalance(positions_df_stk, settled_cash, CHFUSD_quote, account, leverage_live, rebalance_type = 'standard')
                st.stop()
            else:
                st.write("There is no new cash to invest, nor the portfolio needs rebalancing. Stopping.")
                st.stop()
            #######################################################################################################################
            # 3.1 If there is no cash (or after cash was used to buy stocks), check again if the portfolio needs rebalancing.
            #     The user can decide if to rebalance selling stocks or not.

            # Check if a portfolio rebalance is necessary, following the 5/25 rule by Swedroe
            #
            # rule_abs_diff = (allocation_stk['allocation_diff'] > 0.05).any()
            # rule_rel_diff = (allocation_stk['allocation_diff_pct'] > 0.25).any()
            # if rule_abs_diff or rule_rel_diff:
            #     REBALANCE_FLAG = True
            # else:
            #     REBALANCE_FLAG = False
            #
            # if not REBALANCE_FLAG and not BUY_FLAG:
            #     print("I checked if the portfolio needs to be rebalanced using 5/25 rule by Swedroe. \n "
            #           "Rebalance is not necessary and there is no available cash. Exiting.")
            #     ib_exit(ib_client)
            #
            # BUY_SELL_FLAG = False
            # if REBALANCE_FLAG and not BUY_FLAG:
            #     print("I checked if the portfolio needs to be rebalanced using 5/25 rule by Swedroe. \n "
            #           "Rebalance is necessary but there is no available cash. ")
            #     rebalance_user_confirmation = input("Do you want to perform rebalancing by selling stocks? (Yes/No)")
            #     if rebalance_user_confirmation in ['Yes', 'YES', 'Y', 'y']:#
            #         print("I will rebalance the portfolio by selling and buying stocks. ")
            #         BUY_SELL_FLAG = True
            #
            # if BUY_SELL_FLAG:
            #     rebalance(positions_df_stk, settled_cash, rebalance_type='standard')
            #

            # if st.button("Close connection"):
            #     ib_exit(session_state.ib_client)
            #     st.markdown("Connection terminated.")