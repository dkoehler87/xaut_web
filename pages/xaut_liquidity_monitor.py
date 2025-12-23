# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 08:48:57 2025

@author: DKOEH
"""

# -*- coding: utf-8 -*-

import time
import math
import traceback
import pandas as pd
import streamlit as st
import ccxt
import altair as alt
import json
from pathlib import Path

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# st.set_page_config(
#     page_title="XAUT Liquidity Monitor",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# ------------------------------------------------------------
# Config: include BOTH symbol + id (id kept for reference, not displayed)
# NOTE: ccxt unified calls (fetch_order_book) should use `symbol`.
# ------------------------------------------------------------

@st.cache_data
def load_venues():
    root = Path(__file__).resolve().parents[1]  # go up from pages/
    path = root / "venues.json"

    with open(path, "r") as f:
        return json.load(f)

VENUES = load_venues()

# ------------------------------------------------------------
# CCXT init (cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_exchange(ccxt_id: str):
    klass = getattr(ccxt, ccxt_id)
    ex = klass({"enableRateLimit": True})

    # Helps ensure spot orderbooks for exchanges that also have derivatives
    if ccxt_id in {"bybit"}:
        ex.options["defaultType"] = "spot"

    return ex


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


@st.cache_data(ttl=2, show_spinner=False)
def fetch_orderbook_snapshot(ccxt_id: str, symbol: str, limit: int):
    """
    Uses unified symbol. Does NOT call load_markets().
    """
    ex = get_exchange(ccxt_id)
    ob = ex.fetch_order_book(symbol, limit=limit)
    return {
        "timestamp": ob.get("timestamp") or int(time.time() * 1000),
        "bids": ob.get("bids") or [],
        "asks": ob.get("asks") or [],
    }


def iter_levels(levels):
    """
    Yield (price, amount) from an orderbook side, tolerating:
      - [price, amount]
      - [price, amount, ...]
      - (price, amount)
    Skips malformed entries.
    """
    if not levels:
        return
    for lvl in levels:
        try:
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                p = safe_float(lvl[0])
                a = safe_float(lvl[1])
                if math.isfinite(p) and math.isfinite(a):
                    yield p, a
        except Exception:
            continue


def compute_liquidity_metrics_bestpx(orderbook: dict, depth_bps_levels: list[float], depth_unit: str):
    """
    Depth definition:
      - bid depth within X bps of best bid: include bids with p >= best_bid*(1 - X/10000)
      - ask depth within X bps of best ask: include asks with p <= best_ask*(1 + X/10000)

    spread_bps uses mid = (best_bid + best_ask)/2
    depth_unit:
      - "quote": sum(price*amount)
      - "base": sum(amount)
    """
    bids_raw = orderbook.get("bids") or []
    asks_raw = orderbook.get("asks") or []

    best_bid = float("nan")
    best_ask = float("nan")

    for p, _a in iter_levels(bids_raw):
        best_bid = p
        break
    for p, _a in iter_levels(asks_raw):
        best_ask = p
        break

    if not (math.isfinite(best_bid) and math.isfinite(best_ask)) or best_bid <= 0 or best_ask <= 0:
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": float("nan"),
            "spread_bps": float("nan"),
            "depth": {bps: {"bid": float("nan"), "ask": float("nan")} for bps in depth_bps_levels},
        }

    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid) * 10_000.0 if mid > 0 else float("nan")

    def level_value(price, amount):
        return (price * amount) if depth_unit == "quote" else amount

    depth_out = {}
    for bps in depth_bps_levels:
        band = bps / 10_000.0
        bid_cutoff = best_bid * (1.0 - band)
        ask_cutoff = best_ask * (1.0 + band)

        bid_depth = 0.0
        for p, a in iter_levels(bids_raw):
            if p < bid_cutoff:
                break
            bid_depth += level_value(p, a)

        ask_depth = 0.0
        for p, a in iter_levels(asks_raw):
            if p > ask_cutoff:
                break
            ask_depth += level_value(p, a)

        depth_out[bps] = {"bid": bid_depth, "ask": ask_depth}

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": spread_bps,
        "depth": depth_out,
    }


def parse_bps_list(text: str, default=None):
    if default is None:
        default = [25.0, 50.0, 100.0]
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    out = sorted(set(out))
    return out if out else default


def ensure_history_state():
    if "liq_history" not in st.session_state:
        st.session_state.liq_history = []


def prune_history(minutes: int):
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=minutes)
    st.session_state.liq_history = [
        r for r in st.session_state.liq_history
        if pd.to_datetime(r["ts"], utc=True) >= cutoff
    ]


def build_alert_flags(row: dict, spread_alert_bps: float, depth_alert_value: float, depth_band_bps: float, depth_unit: str):
    spread = row.get("spread_bps", float("nan"))
    spread_flag = math.isfinite(spread) and spread > spread_alert_bps

    bid_col = f"bid_depth_{int(depth_band_bps)}bps_{depth_unit}"
    ask_col = f"ask_depth_{int(depth_band_bps)}bps_{depth_unit}"
    bid_depth = row.get(bid_col, float("nan"))
    ask_depth = row.get(ask_col, float("nan"))

    depth_flag = False
    if math.isfinite(bid_depth) and bid_depth < depth_alert_value:
        depth_flag = True
    if math.isfinite(ask_depth) and ask_depth < depth_alert_value:
        depth_flag = True

    return spread_flag, depth_flag


def extract_last_traceback_frame(tb: str):
    lines = (tb or "").strip().splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].lstrip().startswith("File "):
            file_line = lines[idx].strip()
            code_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            return file_line, code_line
    return None, None


def depth_ladder_chart(row: dict, depth_bps_levels: list[float], depth_unit: str, width: int = 300, height: int = 130):
    """
    Mini tornado/butterfly chart:
      - bid depth shown as negative (left)
      - ask depth shown as positive (right)
      - y-axis ordered top→bottom: 25, 50, 100...
      - depth values rendered inside bars
    """
    data = []
    for bps in sorted(depth_bps_levels):
        b = int(bps)
        bid_col = f"bid_depth_{b}bps_{depth_unit}"
        ask_col = f"ask_depth_{b}bps_{depth_unit}"

        bid = row.get(bid_col, float("nan"))
        ask = row.get(ask_col, float("nan"))

        try:
            bid = float(bid)
        except Exception:
            bid = float("nan")
        try:
            ask = float(ask)
        except Exception:
            ask = float("nan")

        if math.isfinite(bid):
            data.append({"band": b, "side": "Bid", "signed": -bid, "abs": bid})
        if math.isfinite(ask):
            data.append({"band": b, "side": "Ask", "signed": ask, "abs": ask})

    df = pd.DataFrame(data)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

    max_abs = float(df["abs"].max())
    if not math.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "band:O",
                sort=sorted([int(b) for b in depth_bps_levels]),
                title="bps",
                axis=alt.Axis(labelAngle=0),
            ),
            x=alt.X(
                "signed:Q",
                scale=alt.Scale(domain=[-max_abs, max_abs]),
                title="Bid | Ask",
                axis=alt.Axis(labels=False, ticks=False, grid=False),
            ),
            color=alt.Color("side:N", legend=None),
        )
        .properties(width=width, height=height)
    )

    labels = (
        alt.Chart(df)
        .mark_text(align="center", baseline="middle", color="white", fontSize=11)
        .encode(
            y=alt.Y("band:O", sort=sorted([int(b) for b in depth_bps_levels])),
            x=alt.X("signed:Q"),
            text=alt.Text("abs:Q", format=",.0f"),
        )
    )

    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
    return bars + labels + zero_rule

def make_rename_map(depth_bps_levels, depth_unit):
    rm = {
        "venue": "Venue",
        "symbol": "Symbol",
        "ts": "Timestamp",
        "best_bid": "Best Bid",
        "best_ask": "Best Ask",
        "spread_bps": "Spread (bps)",
        "alert_depth": "!Depth",
        "alert_spread": "!Spread",
        # keep these if you still want them in history (optional)
        "mid": "Mid",
        "alert_any":"!Any"
    }
    for bps in depth_bps_levels:
        b = int(bps)
        rm[f"bid_depth_{b}bps_{depth_unit}"] = f"Bid Dep. ({b} bps)"
        rm[f"ask_depth_{b}bps_{depth_unit}"] = f"Ask Dep. ({b} bps)"
        
    return rm


def liquidity_page():
    st.title("XAUT Liquidity Monitor")

    # Placeholders so metrics can render directly under title (after we compute data)
    metrics_placeholder = st.container()

    ensure_history_state()

    # ----------------------------
    # Sidebar controls
    # ----------------------------
    st.sidebar.header("Polling")
    selected_venues = st.sidebar.multiselect(
        "Venues",
        options=list(VENUES.keys()),
        default=list(VENUES.keys()),
    )
    refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 2, 60, 20, 5)
    orderbook_limit = st.sidebar.slider("Orderbook levels (per side)", 10, 400, 200, 50)
    charts_per_row = st.sidebar.slider("Charts per row", 1, 4, 3, 1)

    st.sidebar.header("Depth bands")
    depth_bps_levels = parse_bps_list(st.sidebar.text_input("Depth bands (bps)", "25,50,100"))
    depth_unit = st.sidebar.selectbox(
        "Depth unit",
        ["quote", "base"],
        index=0,
        help="quote = sum(price*amount) in USD/USDT terms, base = sum(amount) in XAUT",
    )

    st.sidebar.header("Rolling window")
    window_minutes = st.sidebar.slider("Keep last N minutes", 1, 240, 30, 1)

    st.sidebar.header("Alerts")
    spread_alert_bps = st.sidebar.number_input("Alert if spread (bps) >", min_value=0.0, value=30.0, step=1.0)
    alert_depth_band = st.sidebar.selectbox("Depth band used for alert", options=depth_bps_levels, index=0)

    default_depth_alert = 50_000.0 if depth_unit == "quote" else 5.0
    depth_alert_value = st.sidebar.number_input(
        f"Alert if bid OR ask depth @ {int(alert_depth_band)}bps <",
        min_value=0.0,
        value=float(default_depth_alert),
        step=1000.0 if depth_unit == "quote" else 0.5,
    )

    # Auto refresh (optional dependency)
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_seconds * 1000, key="liq_refresh")
    else:
        st.info("Optional: `pip install streamlit-autorefresh` to auto-poll without manual refresh.")

    # ----------------------------
    # Fetch snapshots
    # ----------------------------
    rows = []
    errors = []

    for venue in selected_venues:
        ccxt_id = VENUES[venue]["ccxt_id"]
        _ = get_exchange(ccxt_id)

        for pair in VENUES[venue]["pairs"]:
            symbol = pair["symbol"]
            snap = None
            try:
                snap = fetch_orderbook_snapshot(ccxt_id, symbol, orderbook_limit)
                metrics = compute_liquidity_metrics_bestpx(snap, depth_bps_levels, depth_unit)

                row = {
                    "venue": venue,
                    "symbol": symbol,
                    "ts": pd.to_datetime(snap["timestamp"], unit="ms", utc=True),
                    "best_bid": metrics["best_bid"],
                    "best_ask": metrics["best_ask"],
                    "mid": metrics["mid"],
                    "spread_bps": metrics["spread_bps"],
                }

                # Round depth fields to 0 dp (stored + displayed)
                for bps in depth_bps_levels:
                    row[f"bid_depth_{int(bps)}bps_{depth_unit}"] = round(metrics["depth"][bps]["bid"], 0)
                    row[f"ask_depth_{int(bps)}bps_{depth_unit}"] = round(metrics["depth"][bps]["ask"], 0)

                spread_flag, depth_flag = build_alert_flags(
                    row, spread_alert_bps, depth_alert_value, alert_depth_band, depth_unit
                )
                row["alert_spread"] = spread_flag
                row["alert_depth"] = depth_flag
                row["alert_any"] = bool(spread_flag or depth_flag)

                rows.append(row)

            except Exception as e:
                bids_len = asks_len = None
                first_bid = first_ask = None
                try:
                    if snap:
                        bids = snap.get("bids", []) or []
                        asks = snap.get("asks", []) or []
                        bids_len = len(bids)
                        asks_len = len(asks)
                        first_bid = bids[0] if bids_len else None
                        first_ask = asks[0] if asks_len else None
                except Exception:
                    pass

                tb = traceback.format_exc()
                file_line, code_line = extract_last_traceback_frame(tb)

                errors.append({
                    "venue": venue,
                    "symbol": symbol,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "where_file_line": file_line,
                    "where_code_line": code_line,
                    "bids_len": bids_len,
                    "asks_len": asks_len,
                    "first_bid": repr(first_bid),
                    "first_ask": repr(first_ask),
                    "traceback": tb,
                })

    # Update rolling history
    if rows:
        st.session_state.liq_history.extend(rows)
        prune_history(window_minutes)

    live_df = pd.DataFrame(rows)
    hist_df = pd.DataFrame(st.session_state.liq_history)

    # ----------------------------
    # Metrics bar directly under title
    # ----------------------------
    with metrics_placeholder:
        # c1, c2, c3 = st.columns(3)
        # c1.metric("Venues monitored", len(selected_venues))
        # c2.metric("Pairs monitored", int(len(live_df)) if not live_df.empty else 0)
        # c3.metric("Active alerts", int(live_df["alert_any"].sum()) if (not live_df.empty and "alert_any" in live_df.columns) else 0)
        
        with st.container():
            cols = st.columns([1, 1, 1, 6])  # last column is spacer
        
            with cols[0]:
                st.metric("Venues monitored", len(selected_venues))
            with cols[1]:
                st.metric("Pairs monitored", int(len(live_df)) if not live_df.empty else 0)
            with cols[2]:
                st.metric(
                    "Active alerts",
                    int(live_df["alert_any"].sum())
                    if (not live_df.empty and "alert_any" in live_df.columns)
                    else 0,
                )

    # ----------------------------
    # Live table
    # ----------------------------
    st.subheader("Live Orderbook Snapshots")

    if not live_df.empty:
        # show last update outside the table
        last_update = pd.to_datetime(live_df["ts"].max(), utc=True) if "ts" in live_df.columns else None
        if last_update is not None:
            st.caption(f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        # Build display columns (NO ts) + move alerts to end
        display_cols = ["venue", "symbol", "best_bid", "best_ask","mid", "spread_bps"]
        for bps in depth_bps_levels:
            display_cols += [
                f"bid_depth_{int(bps)}bps_{depth_unit}",
                f"ask_depth_{int(bps)}bps_{depth_unit}",
            ]
        display_cols += ["alert_depth", "alert_spread"]

        # Create a renamed view for readability
        df_show = live_df[display_cols].copy()

        rename_map = {
            "venue": "Venue",
            "symbol": "Symbol",
            "best_bid": "Best Bid",
            "best_ask": "Best Ask",
            "mid": "Mid",
            "spread_bps": "TOB Spread (bps)",
            "alert_depth": "!Depth",
            "alert_spread": "!Spread",
        }
        for bps in depth_bps_levels:
            b = int(bps)
            rename_map[f"bid_depth_{b}bps_{depth_unit}"] = f"Bid Dep. ({b} bps)"
            rename_map[f"ask_depth_{b}bps_{depth_unit}"] = f"Ask Dep. ({b} bps)"

        df_show = df_show.rename(columns=rename_map)

        # Row highlighting based on !Any
        def highlight_alerts(r):
            if r.get("!Depth", False) or r.get("!Spread", False):
                return ["background-color: rgba(255, 0, 0, 0.15)"] * len(r)
            return [""] * len(r)


        # Formatting
        fmt_map = {
            "Best Bid": "{:,.2f}",
            "Best Ask": "{:,.2f}",
            "Mid": "{:,.2f}",
            "TOB Spread (bps)": "{:,.2f}",
        }
        for bps in depth_bps_levels:
            b = int(bps)
            fmt_map[f"Bid Dep. ({b} bps)"] = "{:,.0f}"
            fmt_map[f"Ask Dep. ({b} bps)"] = "{:,.0f}"

        # Sort for stable viewing
        df_show = df_show.sort_values(["Venue", "Symbol"])

        st.dataframe(
            df_show.style.apply(highlight_alerts, axis=1).format(fmt_map),
            width="content",
            hide_index=True,
        )
    else:
        st.warning("No live data returned (or all venues errored).")

    # ----------------------------
    # Depth ladders
    # ----------------------------
    st.subheader("Depth Ladder Charts")

    if not live_df.empty:
        items = live_df.sort_values(["symbol", "venue"]).to_dict(orient="records")

        for i in range(0, len(items), charts_per_row):
            cols = st.columns(charts_per_row)
            for j, item in enumerate(items[i:i + charts_per_row]):
                with cols[j]:
                    st.markdown(f"**{item['venue']}** — {item['symbol']}")
                    chart = depth_ladder_chart(item, depth_bps_levels, depth_unit, width=300, height=130)
                    st.altair_chart(chart, width="stretch")

                    # Imbalance @ first band (e.g. 25bps)
                    b0 = int(sorted(depth_bps_levels)[0])
                    bid0 = item.get(f"bid_depth_{b0}bps_{depth_unit}", float("nan"))
                    ask0 = item.get(f"ask_depth_{b0}bps_{depth_unit}", float("nan"))
                    if math.isfinite(bid0) and math.isfinite(ask0) and (bid0 + ask0) > 0:
                        imb = (bid0 - ask0) / (bid0 + ask0)
                        st.caption(f"Imbalance @ {b0}bps: {imb:+.2%}")

    # ----------------------------
    # Detailed errors
    # ----------------------------
    if errors:
        with st.expander("Errors (detailed)"):
            for i, err in enumerate(errors, start=1):
                st.markdown(f"### {i}) {err['venue']} — {err['symbol']}")
                st.write(f"**{err['error_type']}**: {err['error_msg']}")

                if err.get("where_file_line"):
                    st.write("**Where it failed**")
                    st.code(f"{err['where_file_line']}\n{err.get('where_code_line','')}", language="text")

                st.write("**Orderbook context**")
                st.json({
                    "bids_len": err.get("bids_len"),
                    "asks_len": err.get("asks_len"),
                    "first_bid": err.get("first_bid"),
                    "first_ask": err.get("first_ask"),
                })

                st.write("**Full traceback**")
                st.code(err["traceback"], language="python")

    # ----------------------------
    # Rolling charts
    # ----------------------------
    st.subheader(f"Rolling Charts (Last {window_minutes} Minutes)")

    if hist_df.empty:
        st.info("No history yet — wait for a couple refresh cycles.")
        return

    pairs = (
        hist_df[["venue", "symbol"]]
        .drop_duplicates()
        .assign(key=lambda d: d["venue"] + " | " + d["symbol"])
        .sort_values("key")["key"]
        .tolist()
    )
    
    ctrl_col, spacer = st.columns([2, 6])  # compact controls on left

    with ctrl_col:
        selected_pair = st.selectbox("Venue/Pair", options=pairs, index=0)
        sel_venue, sel_symbol = [x.strip() for x in selected_pair.split("|", 1)]
        pair_hist = hist_df[(hist_df["venue"] == sel_venue) & (hist_df["symbol"] == sel_symbol)].copy()
        pair_hist = pair_hist.sort_values("ts").copy()
        pair_hist["ts_label"] = pd.to_datetime(pair_hist["ts"], utc=True).dt.strftime("%H:%M:%S")

        
        chart_depth_band = st.selectbox("Depth band (bps)", options=depth_bps_levels, index=0)
        
        bid_col = f"bid_depth_{int(chart_depth_band)}bps_{depth_unit}"
        ask_col = f"ask_depth_{int(chart_depth_band)}bps_{depth_unit}"
    

    spread_chart = (
        alt.Chart(pair_hist)
        .mark_line()
        .encode(
            x=alt.X(
                    "ts_label:O",
                    title="Time (UTC)",
                    axis=alt.Axis(labelOverlap="greedy", labelAngle=-35),
                ),
            y=alt.Y("spread_bps:Q", title="Spread (bps)"),
            tooltip=["ts:T", "spread_bps:Q", "best_bid:Q", "best_ask:Q"],
        )
        .properties(height=220)
    )
    spread_rule = alt.Chart(pd.DataFrame({"y": [spread_alert_bps]})).mark_rule().encode(y="y:Q")
    st.altair_chart(spread_chart + spread_rule, width=900) #width="stretch"

    depth_long = pair_hist.melt(
        id_vars=["ts", "ts_label"],
        value_vars=[bid_col, ask_col],
        var_name="side",
        value_name="depth",
    )
    
    depth_long["side"] = depth_long["side"].map({bid_col: "bid_depth", ask_col: "ask_depth"})
    
    # ensure numeric (extra safety for Arrow/Altair)
    depth_long["depth"] = pd.to_numeric(depth_long["depth"], errors="coerce")
        

    depth_chart = (
        alt.Chart(depth_long)
        .mark_line()
        .encode(
            x=alt.X(
                    "ts_label:O",
                    title="Time (UTC)",
                    axis=alt.Axis(labelOverlap="greedy", labelAngle=-35),
                ),
            y=alt.Y("depth:Q", title=f"Depth @ {int(chart_depth_band)}bps ({depth_unit})"),
            color="side:N",
            tooltip=["ts:T", "side:N", "depth:Q"],
        )
        .properties(height=240)
    )
    depth_rule = alt.Chart(pd.DataFrame({"y": [depth_alert_value]})).mark_rule().encode(y="y:Q")
    st.altair_chart(depth_chart + depth_rule, width=900)


    hist_col, _spacer = st.columns([6, 2])
    with hist_col:
        with st.expander("Show History Data (selected Venue/Pair)"):
            rename_map_hist = make_rename_map(depth_bps_levels, depth_unit)
            
            hist_show = pair_hist.copy()
            hist_show = hist_show.rename(columns=rename_map_hist)
            
            # Optionally drop Mid / !Any / Timestamp if you don't want them:
            hist_show = hist_show.drop(columns=["!Any"], errors="ignore")
            
            st.dataframe(hist_show.tail(300), width="content", hide_index=True)


if __name__ == '__main__':
    liquidity_page()

