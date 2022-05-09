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
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import SessionState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import *
from resources.IBRK_client import IBClient

def app():
    st.title('Connect to Interactive Brokers')

    connected_flg = False
    if st.session_state.ib_client is not None:
        if st.session_state.ib_client.is_authenticated()['authenticated']:
            connected_flg = True
            connected = True

    if connected_flg:
        st.write("Connected to Interactive Brokers.")
        st.stop()
    else:
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

        client_gateway_path = pathlib.Path(r'resources/clientportal.beta.gw').resolve()

        # Create a new session of the IB Web API.
        st.session_state.ib_client = IBClient(
            username=username,
            account=account,
            client_gateway_path=client_gateway_path,
            is_server_running=False
        )

        st.session_state.ib_client._start_server()
        st.session_state.ib_client._server_state(action='save')

        st.markdown("The Interactive Broker server is currently starting up, so we can authenticate your session.  \n" +
                    "- STEP 1: GO TO THE FOLLOWING URL: https://localhost:5000/sso/Login?forwardTo=22&RL=1&ip2loc=on \n"
                    "- STEP 2: LOGIN TO YOUR account WITH YOUR username AND PASSWORD.  \n"
                    "- STEP 3: WHEN YOU SEE `Client login succeeds` RETURN BACK HERE AND CLICK THE BUTTON BELOW TO CHECK IF THE SESSION IS AUTHENTICATED.  \n"
                    )

        with st.form("connect_frm"):
            connect_btn = st.form_submit_button("Confirm connection to IBKR")

        if connect_btn:
            st.write("temp do nothing")
            # connected = st.session_state.ib_client.create_session()
