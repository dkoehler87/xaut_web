# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 12:22:45 2025

@author: DKOEH
"""

import streamlit as st

st.set_page_config(page_title="XAUT Apps", layout="wide")

# Hide Streamlit's default multipage nav section (keeps your own sidebar controls intact)
st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Register pages (URL routing)
home = st.Page(
    "pages/xaut_market_data_viewer.py",
    title="Market Data Viewer",
    url_path="",
    default=True,
)

liq = st.Page(
    "pages/xaut_liquidity_monitor.py",
    title="Liquidity Monitor",
    url_path="liquidity-monitor",
)

nav = st.navigation([home, liq])


def top_nav():
    c1, c2, spacer = st.columns([1.4, 1.4, 6])
    with c1:
        st.page_link(home, label="Market Data Viewer")
    with c2:
        st.page_link(liq, label="Liquidity Monitor")

top_nav()
st.divider()

# Run the selected page and stop here so app.py doesn't render anything else.
nav.run()
st.stop()

