import streamlit as st
import os, sys
import pages.home
# from pages.home import session_state

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
# import SessionState

def app():
    st.title('Advanced Settings')

    st.write('Here you can modify advanced settings for the backtesting.')

    st.markdown('## General parameters')

    st.session_state.DAYS_IN_YEAR = st.number_input("Days in year", min_value=1, max_value=366, value=st.session_state.DAYS_IN_YEAR, step=1,  format='%i', key='DAYS_IN_YEAR_box', help='Number of days in a year. Default is 260.')
    st.session_state.DAYS_IN_YEAR_BOND_PRICE = st.number_input("Days in year for bonds", min_value=1, max_value=366, value=st.session_state.DAYS_IN_YEAR_BOND_PRICE, format='%i', key='DAYS_IN_YEAR_BOND_PRICE_box', help='Number of days in a year used for calculating bond prices from yields. Default is 360.')
    st.session_state.MARGINLOAN_INT_RATE = st.number_input("Interest rate of a margin loan", min_value=0.0, max_value=100.0, value=st.session_state.MARGINLOAN_INT_RATE, format='%f', key='MARGINLOAN_INT_RATE_box', help='Interest rate of a margin loan (annual in percentage)')



    st.markdown('## Strategy parameters')

    col1, col2, col3 = st.columns(3)
    col1.subheader("daily frequency")
    col2.subheader("yearly frequency")
    col3.subheader("custom weights")

    tooltip = 'Number of days (of bars) every which the portfolio is rebalanced. Default is 1 for daily data and 1 for yearly data.'
    st.session_state.reb_period_days=col1.number_input("rebalance months", min_value=1, max_value=36, value=st.session_state.reb_period_days, format='%i', key='reb_period_days_box', help=tooltip)
    st.session_state.reb_period_years=col2.number_input("rebalance years", min_value=1, max_value=5, value=st.session_state.reb_period_years, format='%i', key='reb_period_years_box', help=tooltip)
    st.session_state.reb_period_custweights=col3.number_input("rebalance months", min_value=1, max_value=36, value=st.session_state.reb_period_custweights, format='%i', key='reb_period_custweights_box', help=tooltip)

    tooltip = "Window to calculate the standard deviation of assets returns. Applies to strategy `riskparity` and derived strategies. Default is 20 for daily data and 10 for yearly data."
    st.session_state.lookback_period_short_days=col1.number_input("lookback period volatility", min_value=2, max_value=366, value=st.session_state.lookback_period_short_days, format='%i', key='lookback_period_short_days_box', help=tooltip)
    st.session_state.lookback_period_short_years=col2.number_input("lookback period volatility", min_value=2, max_value=366, value=st.session_state.lookback_period_short_years, format='%i', key='lookback_period_short_years_box', help=tooltip)
    st.session_state.lookback_period_short_custweights=col3.number_input("lookback period volatility", min_value=2, max_value=366, value=st.session_state.lookback_period_short_custweights, format='%i', key='lookback_period_short_custweights_box', help=tooltip)

    tooltip = "Window to calculate the correlation matrix of assets returns. Applies to strategies `riskparity` and derived strategies. Default is 120 for daily data and 10 for yearly data."
    st.session_state.lookback_period_long_days=col1.number_input("lookback period correlation", min_value=2, max_value=366, value=st.session_state.lookback_period_long_days, format='%i', key='lookback_period_long_days_box', help=tooltip)
    st.session_state.lookback_period_long_years=col2.number_input("lookback period correlation", min_value=2, max_value=366, value=st.session_state.lookback_period_long_years, format='%i', key='lookback_period_long_years_box', help=tooltip)
    st.session_state.lookback_period_long_custweights=col3.number_input("lookback period correlation", min_value=2, max_value=366, value=st.session_state.lookback_period_long_custweights, format='%i', key='lookback_period_long_custweights_box', help=tooltip)

    tooltip = "Window to calculate the simple moving average. Applies to strategies `trend_uniform`, `trend_riskparity`, `momentumtrend_uniform`  and `momentumtrend_riskparity`. Default is 252 for daily data and 5 for yearly data."
    st.session_state.moving_average_period_days=col1.number_input("trend period", min_value=2, max_value=366, value=st.session_state.moving_average_period_days, format='%i', key='moving_average_period_days_box', help=tooltip)
    st.session_state.moving_average_period_years=col2.number_input("trend period", min_value=2, max_value=366, value=st.session_state.moving_average_period_years, format='%i', key='moving_average_period_years_box', help=tooltip)
    st.session_state.moving_average_period_custweights=col3.number_input("trend period", min_value=2, max_value=366, value=st.session_state.moving_average_period_custweights, format='%i', key='moving_average_period_custweights_box', help=tooltip)

    tooltip = "Window to calculate the momentum. Applies to strategies `absolutemomentum_uniform`, `relativemomentum_uniform`, `momentumtrend_uniform`, `absolutemomentum_riskparity`, `relativemomentum_riskparity`  and `momentumtrend_riskparity`. Default is 252 for daily data and 5 for yearly data."
    st.session_state.momentum_period_days=col1.number_input("momentum period", min_value=2, max_value=366, value=st.session_state.momentum_period_days, format='%i', key='momentum_period_days_box', help=tooltip)
    st.session_state.momentum_period_years=col2.number_input("momentum period", min_value=2, max_value=366, value=st.session_state.momentum_period_years, format='%i', key='momentum_period_years_box', help=tooltip)
    st.session_state.momentum_period_custweights=col3.number_input("momentum period", min_value=2, max_value=366, value=st.session_state.momentum_period_custweights, format='%i', key='momentum_period_custweights_box', help=tooltip)

    tooltip = "Percentile of assets with the highest return in a period to form the relative momentum portfolio. The higher the percentile, the higher the return quantile."
    st.session_state.momentum_percentile_days=col1.number_input("momentum percentile", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.momentum_percentile_days, format='%f', key='momentum_percentile_days_box', help=tooltip)
    st.session_state.momentum_percentile_years=col2.number_input("momentum percentile", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.momentum_percentile_years, format='%f', key='momentum_percentile_years_box', help=tooltip)
    st.session_state.momentum_percentile_custweights=col3.number_input("momentum percentile", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.momentum_percentile_custweights, format='%f', key='momentum_percentile_custweights_box', help=tooltip)

    tooltip = "Method for the calculation of the correlation matrix. Applies to strategies `riskparity` and `riskparity_nested`. Default is 'pearson'. Alternative is 'spearman'."
    if st.session_state.corrmethod_days=='pearson':
        idx=0
    else:
        idx=1
    st.session_state.corrmethod_days=col1.selectbox("correlation coefficient", ('pearson', 'spearman'), index=idx, key='corrmethod_days_box', help=tooltip)
    if st.session_state.corrmethod_years=='pearson':
        idx=0
    else:
        idx=1
    st.session_state.corrmethod_years=col2.selectbox("correlation coefficient", ('pearson', 'spearman'), index=idx, key='corrmethod_years_box', help=tooltip)
    if st.session_state.corrmethod_custweights=='pearson':
        idx=0
    else:
        idx=1
    st.session_state.corrmethod_custweights=col3.selectbox("correlation coefficient", ('pearson', 'spearman'), index=idx, key='corrmethod_custweights_box', help=tooltip)

    st.markdown('## Report parameters')

    st.session_state.riskfree=st.number_input("risk free rate", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.riskfree, format='%f', key='riskfree_box', help="Risk free rate to be used in metrics like treynor_ratio, sharpe_ratio, etc. Default is 0.01.")
    st.session_state.targetrate=st.number_input("target rate", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.targetrate, format='%f', key='targetrate_box', help="Target return rate to be used in omega_ratio, sortino_ratio, kappa_three_ratio, gain_loss_ratio, upside_potential_ratio. Default is 0.01.")
    st.session_state.alpha=st.number_input("alpha (VaR)", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.alpha, format='%f', key='alpha_box', help="Confidence interval to be used in VaR, CVaR and VaR based metrics (excess VaR, conditional Sharpe Ratio). Default is 0.05.")
    st.session_state.market_mu=st.number_input("market mu", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.market_mu, format='%f', key='market_mu_box', help="Average annual return of the market, to be used in Treynor ratio, Information ratio. Default is 0.07.")
    st.session_state.market_sigma=st.number_input("market sigma", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.market_sigma, format='%f', key='market_sigma_box', help="Annual standard deviation of the market, to be used in Treynor ratio, Information ratio. Default is  0.15.")
    st.session_state.stddev_sample=st.checkbox("standard deviation sample adjustment", value=st.session_state.stddev_sample, key="stddev_sample_box", help='Bessel correction (N-1) when calculating standard deviation from a sample')
    st.session_state.annualize=st.checkbox("annualize first", value=st.session_state.annualize, key="annualize_box", help='calculate annualized metrics by annualizing returns first')
    st.session_state.logreturns=st.checkbox("logreturns", value=st.session_state.logreturns, key="logreturns_box", help='Use logreturns instead of percentage returns when calculating metrics (not recommended). Default is False.')