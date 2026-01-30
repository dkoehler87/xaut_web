# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 12:22:45 2025

@author: DKOEH
"""

import pandas as pd
import streamlit as st
from xaut_data import build_xaut_dataframes, build_xaut_perps_dataframe
from xaut0_data import build_xaut0_dataframes
from usat_data import build_usat_dataframes
import plotly.express as px
import os
import time
import random
import requests


def run_with_cg_backoff(fn, keys: list[str], label: str, max_tries: int = 6):
    """
    Runs a loader fn(api_key) with:
      - key rotation across attempts
      - exponential backoff on HTTP 429
    """
    last_err = None

    for attempt in range(max_tries):
        api_key = pick_coingecko_key(keys, f"cg_rr_{label}")

        try:
            return fn(api_key)

        except requests.HTTPError as e:
            last_err = e
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)

            # Only backoff on 429. Everything else: raise immediately.
            if status != 429:
                raise

            # Respect Retry-After when present, else exponential backoff + jitter
            retry_after = None
            if resp is not None:
                ra = resp.headers.get("Retry-After")
                if ra and ra.isdigit():
                    retry_after = int(ra)

            sleep_s = retry_after if retry_after is not None else min(2 ** attempt, 30) + random.random()
            time.sleep(sleep_s)
            continue

        except Exception as e:
            # Non-HTTP errors (parsing, etc.) shouldn't be retried forever
            raise

    # If we exhausted retries, re-raise the last error
    raise last_err


def _get_secret(name: str) -> str:
    try:
        val = st.secrets.get(name, "")
        return str(val).strip() if val else ""
    except Exception:
        return ""

def _get_env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()

def get_coingecko_api_keys() -> list[str]:
    """
    Returns a list of available CoinGecko API keys (non-empty), in priority order.
    Supports:
      - COINGECKO_API_KEY_1, COINGECKO_API_KEY_2 (recommended)
      - COINGECKO_API_KEY (legacy fallback)
    Works both in Streamlit Secrets and local env vars.
    """
    keys = []

    # Preferred: 2 keys
    for kname in ("COINGECKO_API_KEY_1", "COINGECKO_API_KEY_2"):
        k = _get_secret(kname) or _get_env(kname)
        if k:
            keys.append(k)

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for k in keys:
        if k not in seen:
            deduped.append(k)
            seen.add(k)

    return deduped

def pick_coingecko_key(keys: list[str], counter_name: str) -> str:
    """
    Round-robin selection stored in session_state.
    counter_name lets you have separate rotation streams (e.g., for load vs load2).
    """
    if not keys:
        return ""

    if counter_name not in st.session_state:
        st.session_state[counter_name] = 0

    idx = st.session_state[counter_name] % len(keys)
    st.session_state[counter_name] += 1
    return keys[idx]


def move_col_left_of(df: pd.DataFrame, col: str, ref_col: str) -> pd.DataFrame:
    if col not in df.columns or ref_col not in df.columns:
        return df
    cols = list(df.columns)
    cols.remove(col)
    ref_idx = cols.index(ref_col)
    cols.insert(ref_idx, col)
    return df[cols]


# st.set_page_config(page_title="XAUT Market Viewer", layout="wide")
st.title("Market Data Viewer - Coingecko")


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
    refresh = st.button("Refresh data")
    st.markdown("---")
    st.subheader("Quick Filters")
    tp_search = st.text_input("Trading pair contains", value="")
    venue_search = st.text_input("Venue contains", value="")
    venue_type_filter = st.multiselect("Venue type", ["cex", "dex"], default=[])

    st.markdown("---")
    st.subheader("Numeric Filters")
    min_usd_vol = st.number_input("Min USD volume", value=0.0, min_value=0.0)
    max_spread = st.number_input("Max TOB spread (bps)", value=10_000.0, min_value=0.0)

#Retrive the Coingecko API Keys
coingecko_keys = get_coingecko_api_keys()

if not coingecko_keys:
    st.warning("No CoinGecko API key found. Set COINGECKO_API_KEY_1/2 in Secrets or env vars.")

@st.cache_data(ttl=600, show_spinner=False)   # 10 minutes
def load(api_key: str):
    return build_xaut_dataframes(coingecko_api_key=api_key)

@st.cache_data(ttl=600, show_spinner=False)
def load2(api_key: str):
    return build_xaut0_dataframes(coingecko_api_key=api_key)

@st.cache_data(ttl=600, show_spinner=False)
def load3(api_key: str):
    return build_usat_dataframes(coingecko_api_key=api_key)

@st.cache_data(ttl=600, show_spinner=False)
def load_xaut_perps(api_key: str):
    return build_xaut_perps_dataframe(coingecko_api_key=api_key)


if refresh:
    st.cache_data.clear()
    st.session_state.pop("data_bundle", None)

try:
    if "data_bundle" not in st.session_state:
        with st.spinner("Loading data..."):
            cex_df, dex_df, usdt_df, btc_df, usd_df, final_df = run_with_cg_backoff(load,  coingecko_keys, "xaut")
            xaut0_df = run_with_cg_backoff(load2, coingecko_keys, "xaut0")
            usat_df  = run_with_cg_backoff(load3, coingecko_keys, "usat")
            xaut_perps_df = run_with_cg_backoff(load_xaut_perps, coingecko_keys, "xaut_perps")

        st.session_state["data_bundle"] = (
            cex_df, dex_df, usdt_df, btc_df, usd_df, final_df, xaut0_df, usat_df, xaut_perps_df
        )

    (cex_df, dex_df, usdt_df, btc_df, usd_df, final_df, xaut0_df, usat_df, xaut_perps_df) = st.session_state["data_bundle"]

except Exception as e:
    st.error("App crashed while loading data. Here is the exception:")
    st.exception(e)
    st.stop()


# --- Token selector (3 buttons) ---
token = st.segmented_control(
    "Token",
    options=["XAUT", "XAUT0", "USAT"],
    default="XAUT",
)

market_type = st.segmented_control(
    "Market",
    options=["Spot", "Perpetuals"],
    default="Spot",
)


if market_type == "Spot":
    token_to_df = {
        "XAUT": final_df,
        "XAUT0": xaut0_df,
        "USAT": usat_df,
    }
else:
    token_to_df = {
        "XAUT": xaut_perps_df,
        "XAUT0": xaut0_df.iloc[0:0],  # empty
        "USAT": usat_df.iloc[0:0],    # empty
    }

base_df = token_to_df[token]

if market_type == "Perpetuals" and token != "XAUT":
    st.info("Perpetuals are only available for XAUT via CoinGecko.")


# ----------------------------
# Exclusion List (separate per Market: Spot vs Perpetuals)
# ----------------------------
if "excluded_venues_spot" not in st.session_state:
    st.session_state["excluded_venues_spot"] = []
if "excluded_venues_perps" not in st.session_state:
    st.session_state["excluded_venues_perps"] = []

if market_type == "Spot":
    # Spot venue universe ONLY (do NOT include perps venues)
    all_venues = sorted(
        set(
            pd.concat(
                [
                    final_df.get("Venue", pd.Series(dtype=str)).dropna().astype(str),
                    xaut0_df.get("Venue", pd.Series(dtype=str)).dropna().astype(str),
                    usat_df.get("Venue", pd.Series(dtype=str)).dropna().astype(str),
                ],
                ignore_index=True
            ).unique().tolist()
        )
    )
    widget_key = "excluded_venues_spot"
else:
    # Perps venue universe ONLY
    all_venues = sorted(
        set(
            xaut_perps_df.get("Venue", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        )
    )
    widget_key = "excluded_venues_perps"

# If the universe changes (e.g., refresh), prune selections that no longer exist
allowed = set(all_venues)
st.session_state[widget_key] = [v for v in st.session_state[widget_key] if v in allowed]

def clear_exclusions_for_current_market():
    st.session_state[widget_key] = []

st.multiselect(
    "Exclusion List",
    options=all_venues,
    key=widget_key,
    help="Selections are preserved separately for Spot and Perpetuals.",
)

st.button("Clear exclusions", on_click=clear_exclusions_for_current_market)


def quote_ccy_from_pair(pair: str) -> str:
    """
    Extract quote currency from Trading Pair strings like:
      XAUT/USDT, XAUT-USDT, XAUT_USDT, etc.
    Falls back gracefully if format is unexpected.
    """
    if not isinstance(pair, str):
        return ""
    p = pair.strip().upper()
    for sep in ("/", "-", "_", ":"):
        if sep in p:
            parts = p.split(sep)
            if len(parts) >= 2:
                return parts[-1].strip()
    return ""

def fmt_pct_trim_6(x):
    try:
        if pd.isna(x):
            return ""
        return f"{round(float(x), 6):.6f}".rstrip("0").rstrip(".") + "%"
    except Exception:
        return ""


def breakdown_df(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    out = df

    # CEX/DEX buckets (based on Venue Type)
    if bucket == "CEX":
        if "Venue Type" in out.columns:
            return out[out["Venue Type"].astype(str).str.lower() == "cex"]
        return out.iloc[0:0]

    if bucket == "DEX":
        if "Venue Type" in out.columns:
            return out[out["Venue Type"].astype(str).str.lower() == "dex"]
        return out.iloc[0:0]

    # Quote currency buckets (based on Trading Pair)
    if bucket in ("USDT", "BTC", "USD"):
        if "Trading Pair" not in out.columns:
            return out.iloc[0:0]
        q = out["Trading Pair"].astype(str).map(quote_ccy_from_pair)
        return out[q == bucket]

    # ALL
    return out

def fmt_usd(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "$0"

def apply_quick_filters(df: pd.DataFrame, excluded_venues: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()

    if excluded_venues and "Venue" in out.columns:
        excl = set(str(v) for v in excluded_venues)
        out = out[~out["Venue"].astype(str).isin(excl)]

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

    st.plotly_chart(fig, width="stretch")

breakdowns = ["ALL", "CEX", "DEX", "USDT", "BTC", "USD"]
tabs = st.tabs(breakdowns)

for tab, name in zip(tabs, breakdowns):
    with tab:
        st.subheader(f"{token} — {name}")
        df = breakdown_df(base_df, name)

        active_exclusions = st.session_state["excluded_venues_spot"] if market_type == "Spot" else st.session_state["excluded_venues_perps"]
        filtered = apply_quick_filters(df, excluded_venues=active_exclusions)

        # Recompute market share based on filtered view
        filtered = filtered.copy()
        filtered["Volume (USD)"] = pd.to_numeric(filtered["Volume (USD)"], errors="coerce")
        total_usd = filtered["Volume (USD)"].sum(skipna=True)
        filtered["Market Share"] = (filtered["Volume (USD)"] / total_usd) if total_usd and total_usd > 0 else 0.0
        
        filtered = move_col_left_of(filtered, "Venue Type", "Market Share")

        # metrics (same as you already do)
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
                
        # Perpetuals-specific formatting
        for col in ["Open Interest", "Index"]:
            if col in formatted.columns:
                format_dict[col] = "{:,.2f}"

        # Add Funding Rate formatter into the same formatter dict (DON'T call .format() twice)
        if "Funding Rate" in formatted.columns:
            format_dict["Funding Rate"] = fmt_pct_trim_6
        
        styler = formatted.style.format(format_dict)
        
        if "Trust Score" in formatted.columns:
            styler = styler.map(trust_score_style, subset=["Trust Score"])

        st.dataframe(styler, width="stretch", hide_index=True)

        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{token.lower()}_{name.lower()}_filtered.csv",
            mime="text/csv",
        )

        st.markdown("### Market Share (by venue)")
        render_market_share_pie_plotly(
            filtered,
            label_col="Venue",
            top_n=10,
            title=f"{token} {name} Market Share by Venue"
        )
