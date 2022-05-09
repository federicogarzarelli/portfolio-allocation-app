import streamlit as st
from app.multiapp import MultiApp
from app.pages import home, settings, exploreDB, signals, IBKR, IBKR_connection # import your app modules here
from PIL import Image
import utils
import os, sys
import resources

app = MultiApp()

wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logo_path = utils.find('logo_small.png', wd)
# #print(logo_path)
logo = Image.open(logo_path)
#
# header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(utils.img_to_bytes(logo_path))
# st.markdown(header_html, unsafe_allow_html=True)
#
st.image(logo)
#
# utils.delete_output_first()

st.markdown("""
# Backtest and live trading
This app backtests portfolio allocation strategies using historical data; provides live market signals for some strategies and 
allows to automatically trade the strats.   
""")

# Add all your application here
app.add_app("Backtest", home.app)
app.add_app("Backtest Advanced Settings", settings.app)
app.add_app("Explore Prices DB", exploreDB.app)
app.add_app("Market signals", signals.app)
app.add_app("IBKR connection", IBKR_connection.app)
app.add_app("IBKR automation", IBKR.app)

# The main app
app.run()

