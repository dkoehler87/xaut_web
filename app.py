# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 12:22:45 2025

@author: DKOEH
"""

import streamlit as st

st.set_page_config(page_title="Volume & Liquidity Monitor", layout="wide")

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

liq_usat = st.Page(
    "pages/usat_liquidity_monitor.py",
    title="Liquidity Monitor - USAT",
    url_path="liquidity-monitor-usat",
)

hist = st.Page(
    "pages/historical_data.py",
    title="Historical Data",
    url_path="historical-data",
)

xaut_gold = st.Page(
    "pages/xaut_vs_gold.py",
    title="XAUT vs. Gold",
    url_path="xaut-vs-gold",
)

market_lookup = st.Page(
    "pages/market_lookup.py",
    title="Market Lookup",
    url_path="market-lookup",
)

usdt_pricing = st.Page(
    "pages/usdt_pricing.py",
    title="USDT Pricing",
    url_path="usdt-pricing",
)

nav = st.navigation([home, liq, liq_usat, hist, xaut_gold, market_lookup, usdt_pricing])


def top_nav():
    c1, c2, c3, c4, c5, c6, c7, spacer = st.columns([1.5, 1.7, 1.9, 1.5, 1.3, 1.5, 1.2, 1.9])
    with c1:
        st.page_link(home, label="Market Data Viewer")
    with c2:
        st.page_link(liq, label="Liquidity Monitor - XAUT")
    with c3:
        st.page_link(liq_usat, label="Liquidity Monitor - USAT")
    with c4:
        st.page_link(hist, label="Historical Data")
    with c5:
        st.page_link(xaut_gold, label="XAUT vs. Gold")
    with c6:
        st.page_link(market_lookup, label="Market Lookup")
    with c7:
        st.page_link(usdt_pricing, label="USDT Pricing")

top_nav()
st.divider()

# Run the selected page and stop here so app.py doesn't render anything else.
nav.run()
st.stop()

