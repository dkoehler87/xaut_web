# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 10:33:24 2025

@author: DKOEH
"""

import pandas as pd
import streamlit as st
from xaut_data import build_xaut_dataframes
import plotly.express as px


st.set_page_config(page_title="XAUT Market Viewer", layout="wide")
st.title("XAUT (Tether Gold) – Market Data Viewer")


DECIMAL_2_COLS = [
    "Last",
    "TOB Spread (bps)",
]

DECIMAL_0_COLS = [
    "Volume",
    "Volume (USD)",
    "Bid Depth (200 bps)",
    "Ask Depth (200 bps)",
]

TRUST_COLORS = {
    "green": "#1e8e3e",   # strong green
    "yellow": "#f9ab00",  # strong yellow
    "red": "#d93025",     # strong red
}

PCT_0_COLS = ["Market Share"]



with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("CoinGecko API Key", value="", type="password")
    st.caption("Leave blank if you don’t need a key for your usage tier.")
    refresh = st.button("Refresh data")
    st.markdown("---")
    st.subheader("Quick Filters (apply within current tab)")
    tp_search = st.text_input("Trading pair contains", value="")
    venue_search = st.text_input("Venue contains", value="")
    venue_type_filter = st.multiselect("Venue type", ["cex", "dex"], default=[])

    st.markdown("---")
    st.subheader("Numeric Filters")
    min_usd_vol = st.number_input("Min USD volume", value=0.0, min_value=0.0)
    max_spread = st.number_input("Max TOB spread (bps)", value=10_000.0, min_value=0.0)


@st.cache_data(ttl=60, show_spinner=False)
def load(api_key: str):
    return build_xaut_dataframes(coingecko_api_key=api_key)


if refresh:
    st.cache_data.clear()

try:
    with st.spinner("Loading data..."):
        cex_df, dex_df, usdt_df, btc_df, usd_df, final_df = load(api_key)
except Exception as e:
    st.error("App crashed while loading data. Here is the exception:")
    st.exception(e)
    st.stop()


tabs = st.tabs(["ALL","CEX", "DEX", "USDT", "BTC", "USD"])

tab_map = {
    "ALL": final_df,
    "CEX": cex_df,
    "DEX": dex_df,
    "USDT": usdt_df,
    "BTC": btc_df,
    "USD": usd_df
}

def fmt_usd(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "$0"

def apply_quick_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if tp_search.strip():
        out = out[out["Trading Pair"].astype(str).str.contains(tp_search, case=False, na=False)]

    if venue_search.strip():
        out = out[out["Venue"].astype(str).str.contains(venue_search, case=False, na=False)]

    if venue_type_filter:
        out = out[out["Venue Type"].astype(str).isin(venue_type_filter)]

    # numeric filters
    out["Volume (USD)"] = pd.to_numeric(out["Volume (USD)"], errors="coerce")
    out["TOB Spread (bps)"] = pd.to_numeric(out["TOB Spread (bps)"], errors="coerce")

    out = out[out["Volume (USD)"].fillna(0) >= float(min_usd_vol)]
    out = out[out["TOB Spread (bps)"].fillna(0) <= float(max_spread)]

    return out


def format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in DECIMAL_2_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    for col in DECIMAL_0_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)

    return out


def trust_score_style(val):
    if pd.isna(val):
        return ""

    color = TRUST_COLORS.get(str(val).lower())
    if not color:
        return ""

    return (
        f"background-color: {color}; "
        f"color: black; "
        f"font-weight: 600;"
    )

def render_market_share_pie_plotly(
    df: pd.DataFrame,
    label_col: str = "Venue",
    top_n: int = 10,
    title: str = "Market Share"
):
    if df.empty:
        st.info("No data for market share chart.")
        return

    if "Market Share" not in df.columns:
        st.info("market_share column not found.")
        return

    plot_df = df.copy()

    plot_df[label_col] = plot_df[label_col].astype(str)
    plot_df["Market Share"] = pd.to_numeric(plot_df["Market Share"], errors="coerce").fillna(0)

    agg = (
        plot_df.groupby(label_col, dropna=False)["Market Share"]
        .sum()
        .sort_values(ascending=False)
    )

    if agg.sum() <= 0:
        st.info("Market share total is 0.")
        return

    if len(agg) > top_n:
        top = agg.head(top_n)
        other = agg.iloc[top_n:].sum()
        agg = pd.concat([top, pd.Series({"Other": other})])

    pie_df = agg.reset_index()
    pie_df.columns = [label_col, "Market Share"]

    fig = px.pie(
        pie_df,
        names=label_col,
        values="Market Share",
        title=title,
        hole=0.35,  # donut style
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    )

    fig.update_layout(
        showlegend=True,
        margin=dict(t=40, b=0, l=0, r=0),
    )

    st.plotly_chart(fig, use_container_width=True)



for tab, (name, df) in zip(tabs, tab_map.items()):
    with tab:
        st.subheader(name)

        filtered = apply_quick_filters(df)
        
        # Recompute market share based on *filtered view* so the chart & column match what’s displayed
        filtered = filtered.copy()
        filtered["Volume (USD)"] = pd.to_numeric(filtered["Volume (USD)"], errors="coerce")
        total_usd = filtered["Volume (USD)"].sum(skipna=True)
        filtered["Market Share"] = (filtered["Volume (USD)"] / total_usd) if total_usd and total_usd > 0 else 0.0


        # Make sure usd_volume is numeric
        df_usd = df.copy()
        df_usd["Volume (USD)"] = pd.to_numeric(df_usd["Volume (USD)"], errors="coerce").fillna(0)
        
        filtered_usd = filtered.copy()
        filtered_usd["Volume (USD)"] = pd.to_numeric(filtered_usd["Volume (USD)"], errors="coerce").fillna(0)
        
        total_usd_volume = float(df_usd["Volume (USD)"].sum())
        filtered_usd_volume = float(filtered_usd["Volume (USD)"].sum())
        
        c1, c2, c3, c4 = st.columns([1, 1, 1.4, 2])
        with c1:
            st.metric("Rows (filtered)", f"{len(filtered):,}")
        with c2:
            st.metric("Rows (total)", f"{len(df):,}")
        with c3:
            st.metric("USD Volume (filtered)", fmt_usd(filtered_usd_volume))
        with c4:
            st.metric("USD Volume (total)", fmt_usd(total_usd_volume))


        formatted = format_numeric_columns(filtered)
        
        format_dict = {}
        
        for col in DECIMAL_2_COLS:
            if col in formatted.columns:
                format_dict[col] = "{:,.2f}"
        
        for col in DECIMAL_0_COLS:
            if col in formatted.columns:
                format_dict[col] = "{:,.0f}"
                
        for col in PCT_0_COLS:
            if col in formatted.columns:
                format_dict[col] = "{:.0%}"

        styler = (
            formatted.style
            .format(format_dict)
            .applymap(trust_score_style, subset=["Trust Score"])
        )
        
        st.dataframe(
            styler,
            use_container_width=True,
            hide_index=True,
        )
        
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"xaut_{name.lower()}_filtered.csv",
            mime="text/csv",
        )
        
        st.markdown("### Market Share (by venue)")
        render_market_share_pie_plotly(
            filtered,
            label_col="Venue",
            top_n=10,
            title=f"{name} Market Share by Venue"
        )











