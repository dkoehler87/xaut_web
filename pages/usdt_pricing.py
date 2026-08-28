from __future__ import annotations

import os
from typing import Any

import ccxt
import altair as alt
import pandas as pd
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

COINGECKO_TICKERS_URL = "https://api.coingecko.com/api/v3/coins/tether/tickers"
MAX_PAGES = 5
APPROVED_EXCHANGE_NAMES = (
    "Binance", "OKX", "Coinbase Exchange", "MEXC", "Gate.io", "KuCoin",
    "Bybit", "Bitget", "Kraken", "Bitstamp by Robinhood", "Crypto.com Exchange",
    "WhiteBIT", "Bitfinex",
)
ALLOWED_OTHER_VENUES = {
    "binance", "okx", "coinbase exchange", "coinbase", "mexc", "gate.io",
    "gate", "kucoin", "bybit", "bitget", "kraken", "bitstamp",
    "bitstamp by robinhood", "crypto.com exchange", "crypto.com", "whitebit",
    "bitfinex",
}
CCXT_EXCHANGE_IDS = {
    "binance": "binance",
    "okx": "okx",
    "coinbase exchange": "coinbase",
    "coinbase": "coinbase",
    "mexc": "mexc",
    "gate.io": "gate",
    "gate": "gate",
    "kucoin": "kucoin",
    "bybit": "bybit",
    "bitget": "bitget",
    "kraken": "kraken",
    "bitstamp": "bitstamp",
    "bitstamp by robinhood": "bitstamp",
    "crypto.com exchange": "cryptocom",
    "crypto.com": "cryptocom",
    "whitebit": "whitebit",
    "bitfinex": "bitfinex",
}
DISPLAY_COLUMNS = [
    "Rank", "Exchange", "Pair", "Last", "Implied USDT Price (USD)",
    "Deviation (bps)", "24h Volume (USD)", "Bid Depth (2%)",
    "Ask Depth (2%)", "Last Traded At",
]


def get_coingecko_api_key() -> str:
    for name in ("COINGECKO_API_KEY_1", "COINGECKO_API_KEY_2", "COINGECKO_API_KEY"):
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        value = value or os.getenv(name, "")
        if value:
            return str(value)
    return ""


@st.cache_data(ttl=21_600, show_spinner=False)
def fetch_usdt_tickers(api_key: str) -> tuple[list[dict[str, Any]], str | None]:
    headers = {"x-cg-demo-api-key": api_key} if api_key else {}
    tickers: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        response = requests.get(
            COINGECKO_TICKERS_URL,
            headers=headers,
            params={"page": page, "order": "volume_desc", "depth": "true"},
            timeout=30,
        )
        if response.status_code == 429:
            return tickers, "CoinGecko rate limit reached; showing the pages received so far."
        response.raise_for_status()
        page_tickers = response.json().get("tickers", [])
        if not page_tickers:
            break
        tickers.extend(page_tickers)
        if len(page_tickers) < 100:
            break
    return tickers, None


def safe_number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_ticker_dataframe(tickers: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        base = str(ticker.get("base") or "").upper()
        quote = str(ticker.get("target") or "").upper()
        last = safe_number(ticker.get("last"))
        implied = None
        if last is not None and last > 0:
            if base == "USDT" and quote in {"USD", "USDC"}:
                implied = last
            elif base == "USDC" and quote == "USDT":
                implied = 1 / last

        market = ticker.get("market") or {}
        rows.append({
            "Exchange": market.get("name"),
            "Base": base,
            "Quote": quote,
            "Pair": f"{base}/{quote}",
            "Last": last,
            "Implied USDT Price (USD)": implied,
            "Deviation (bps)": ((implied - 1) * 10_000 if implied is not None else None),
            "24h Volume (USD)": safe_number((ticker.get("converted_volume") or {}).get("usd")),
            "Bid Depth (2%)": safe_number(ticker.get("cost_to_move_down_usd")),
            "Ask Depth (2%)": safe_number(ticker.get("cost_to_move_up_usd")),
            "Last Traded At": ticker.get("last_traded_at"),
            "Is Anomaly": ticker.get("is_anomaly"),
            "Is Stale": ticker.get("is_stale"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Exchange Normalized"] = df["Exchange"].fillna("").str.strip().str.casefold()
    return df


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values("24h Volume (USD)", ascending=False, na_position="last").copy()


def quote_depth(levels: list, mid: float, bps: int, side: str) -> float:
    cutoff = mid * (1 - bps / 10_000) if side == "bid" else mid * (1 + bps / 10_000)
    total = 0.0
    for level in levels:
        if len(level) < 2:
            continue
        price = safe_number(level[0])
        amount = safe_number(level[1])
        if price is None or amount is None:
            continue
        inside_band = price >= cutoff if side == "bid" else price <= cutoff
        if inside_band:
            total += price * amount
    return total


def base_asset_depth(levels: list, mid: float, bps: int, side: str) -> float:
    """Sum base units inside a price band."""
    cutoff = mid * (1 - bps / 10_000) if side == "bid" else mid * (1 + bps / 10_000)
    total = 0.0
    for level in levels:
        if len(level) < 2:
            continue
        price = safe_number(level[0])
        amount = safe_number(level[1])
        if price is None or amount is None:
            continue
        inside_band = price >= cutoff if side == "bid" else price <= cutoff
        if inside_band:
            total += amount
    return total


def resolve_symbol(exchange: ccxt.Exchange, base: str, quote: str) -> str | None:
    preferred = f"{base}/{quote}"
    if preferred in exchange.markets:
        return preferred
    for symbol, market in exchange.markets.items():
        if market.get("base") == base and market.get("quote") == quote and market.get("spot"):
            return symbol
    return None


@st.cache_data(ttl=5, show_spinner=False)
def fetch_live_cross_leg(exchange_id: str, base: str, quote: str) -> dict[str, Any]:
    """Fetch one live ticker/order book for the user-selected cross monitor."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True, "timeout": 15_000})
        exchange.load_markets()
        symbol = resolve_symbol(exchange, base, quote)
        if not symbol:
            return {"Status": "Pair unavailable in CCXT"}

        ticker = exchange.fetch_ticker(symbol)
        order_book = exchange.fetch_order_book(symbol)
        bids = order_book.get("bids") or []
        asks = order_book.get("asks") or []
        best_bid = safe_number(bids[0][0]) if bids else None
        best_ask = safe_number(asks[0][0]) if asks else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        last = safe_number(ticker.get("last"))

        if exchange_id == "bitfinex":
            raw_info = ticker.get("info")
            raw_last = (
                safe_number(raw_info[6])
                if isinstance(raw_info, (list, tuple)) and len(raw_info) > 6
                else None
            )
            if raw_last is not None:
                last = raw_last

        status = "OK"
        if mid and (last is None or last <= 0 or not 0.5 <= last / mid <= 1.5):
            last = mid
            status = "Ticker last invalid; using BBO mid"

        usdt_depth = None
        if quote == "USDT" and mid:
            # Bid orders hold USDT and are available to purchase the base asset.
            usdt_depth = quote_depth(bids, mid, 200, "bid")
        elif base == "USDT" and mid:
            # Ask orders offer USDT itself; sum base units rather than USD/base depth.
            usdt_depth = base_asset_depth(asks, mid, 200, "ask")

        return {
            "Pair": symbol,
            "Last": last,
            "USDT Depth (2%)": usdt_depth,
            "Observed At": pd.Timestamp.utcnow(),
            "Status": status,
        }
    except Exception as exc:
        message = str(exc)
        return {
            "Observed At": pd.Timestamp.utcnow(),
            "Status": message[:180] + ("..." if len(message) > 180 else ""),
        }


def selector_options(candidates: pd.DataFrame) -> list[dict[str, str]]:
    columns = ["Exchange", "Exchange Normalized", "Base", "Quote", "Pair"]
    if candidates.empty:
        return []
    unique = candidates.drop_duplicates(["Exchange Normalized", "Base", "Quote"])
    return unique[columns].to_dict("records")


def option_label(option: dict[str, str]) -> str:
    return f"{option['Exchange']} — {option['Pair']}"


def build_top_markets(candidates: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    result = candidates.head(limit).copy()
    if result.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result[DISPLAY_COLUMNS]


def style_markets(df: pd.DataFrame):
    formats = {
        "Last": "{:,.8f}",
        "Implied USDT Price (USD)": "${:,.6f}",
        "Deviation (bps)": "{:+,.2f}",
        "24h Volume (USD)": "${:,.0f}",
        "Bid Depth (2%)": "${:,.0f}",
        "Ask Depth (2%)": "${:,.0f}",
    }
    styler = df.style.format(formats, na_rep="—")
    if "Implied USDT Price (USD)" in df.columns:
        styler = styler.set_properties(
            subset=["Implied USDT Price (USD)"],
            **{
                "background-color": "#eaf4ff",
                "color": "#0f172a",
                "font-weight": "700",
                "border-left": "1px solid #93c5fd",
                "border-right": "1px solid #93c5fd",
            },
        )
        implied_col_index = list(df.columns).index("Implied USDT Price (USD)")
        styler = styler.set_table_styles(
            [{
                "selector": f"th.col_heading.level0.col{implied_col_index}",
                "props": [
                    ("background-color", "#dbeafe"),
                    ("color", "#0f172a"),
                    ("font-weight", "700"),
                ],
            }],
            overwrite=False,
        )
    return styler


def render_market_table(title: str, caption: str, df: pd.DataFrame) -> None:
    st.subheader(title)
    st.caption(caption)
    if df.empty:
        st.info("No qualifying markets were returned by CoinGecko.")
    else:
        st.dataframe(style_markets(df), width="stretch", hide_index=True)


st.title("USDT Pricing")
st.caption(
    "CoinGecko supplies six-hour market rankings, volume, and ±2% table depth; "
    "CCXT is used only after the two live-cross markets are selected."
)

with st.sidebar:
    st.header("USDT Pricing")
    live_auto_refresh = st.toggle("Live chart auto-refresh", value=True)
    refresh_interval_seconds = st.select_slider(
        "CCXT refresh interval",
        options=[5, 10, 15, 30, 60],
        value=5,
        format_func=lambda seconds: f"{seconds} seconds",
        disabled=not live_auto_refresh,
    )
    if st.button("Refresh now", width="stretch"):
        fetch_live_cross_leg.clear()
        st.rerun()
    st.caption(
        f"The selected live cross refreshes every {refresh_interval_seconds} seconds. "
        "CoinGecko rankings and table depth are cached for 6 hours."
    )
    with st.expander("Approved exchanges", expanded=True):
        for exchange_name in APPROVED_EXCHANGE_NAMES:
            st.markdown(f"- {exchange_name}")
    st.caption("Each table is limited to three markets to control CCXT requests.")

try:
    with st.spinner("Loading USDT markets from CoinGecko..."):
        raw_tickers, warning = fetch_usdt_tickers(get_coingecko_api_key())
except requests.RequestException as exc:
    st.error(f"CoinGecko request failed: {exc}")
    st.stop()

if warning:
    st.warning(warning)

all_markets = build_ticker_dataframe(raw_tickers)
if all_markets.empty:
    st.warning("CoinGecko returned no USDT markets.")
    st.stop()

fresh_markets = all_markets[
    ~all_markets["Is Anomaly"].fillna(False)
    & ~all_markets["Is Stale"].fillna(False)
    & all_markets["Exchange Normalized"].isin(ALLOWED_OTHER_VENUES)
].copy()

usdt_usd_candidates = rank_candidates(
    fresh_markets[(fresh_markets["Base"] == "USDT") & (fresh_markets["Quote"] == "USD")],
)
stable_cross_candidates = rank_candidates(fresh_markets[
    ((fresh_markets["Base"] == "USDC") & (fresh_markets["Quote"] == "USDT"))
    | ((fresh_markets["Base"] == "USDT") & (fresh_markets["Quote"] == "USDC"))
])
other_usdt_candidates = rank_candidates(fresh_markets[
    (fresh_markets["Quote"] == "USDT")
    & (fresh_markets["Base"] != "USDC")
])

usdt_usd = build_top_markets(usdt_usd_candidates, limit=3)
stable_crosses = build_top_markets(stable_cross_candidates, limit=3)
other_usdt = build_top_markets(other_usdt_candidates, limit=3)

metric_cols = st.columns(3)
metric_cols[0].metric("USDT/USD sources", len(usdt_usd))
metric_cols[1].metric("USDC/USDT sources", len(stable_crosses))
metric_cols[2].metric("Approved BASE/USDT markets", len(other_usdt))

render_market_table("Top 3 USDT/USD Markets", "Approved venues only. Price, volume, and ±2% depth are from CoinGecko.", usdt_usd)
render_market_table("Top 3 USDC/USDT and USDT/USDC Markets", "Approved venues only. Either pair orientation may appear.", stable_crosses)
render_market_table(
    "Top 3 BASE/USDT Markets",
    "USDC pairs excluded; approved venues only.",
    other_usdt.drop(columns=["Implied USDT Price (USD)", "Deviation (bps)"]),
)

st.caption(
    "Summary-table depth is CoinGecko's reported USD cost to move the market by ±2%. "
    "Implied USDT/USD is shown only for direct USD and USDC crosses."
)


# -----------------------------------------------------------------------------
# User-selected live two-market cross
# -----------------------------------------------------------------------------
st.divider()
st.header("Live BASE/USDT Cross Monitor")
st.caption(
    "Choose one BASE/USDT market and one USD or USDC reference market. "
    f"Only these two selections request live CCXT order books every {refresh_interval_seconds} seconds."
)

base_options = selector_options(other_usdt_candidates)
reference_options = selector_options(
    rank_candidates(pd.concat([usdt_usd_candidates, stable_cross_candidates], ignore_index=True))
)

if not base_options or not reference_options:
    st.info("No eligible market pair selections are available.")
else:
    selection_cols = st.columns(2)
    with selection_cols[0]:
        selected_base = st.selectbox(
            "BASE/USDT market",
            options=base_options,
            format_func=option_label,
        )
    with selection_cols[1]:
        selected_reference = st.selectbox(
            "USDT reference market",
            options=reference_options,
            format_func=option_label,
        )

    start_live_monitor = st.toggle(
        "Start live monitor",
        value=False,
        help=(
            "CCXT requests and automatic refreshes at the configured sidebar interval "
            "begin only when this is enabled."
        ),
    )
    if not start_live_monitor:
        st.info("Select the two markets, then enable **Start live monitor** to begin CCXT updates.")
        st.stop()

    selection_id = "|".join([
        selected_base["Exchange Normalized"], selected_base["Base"], selected_base["Quote"],
        selected_reference["Exchange Normalized"], selected_reference["Base"], selected_reference["Quote"],
    ])
    if st.session_state.get("usdt_cross_selection") != selection_id:
        st.session_state["usdt_cross_selection"] = selection_id
        st.session_state["usdt_cross_history"] = []

    base_live = fetch_live_cross_leg(
        CCXT_EXCHANGE_IDS[selected_base["Exchange Normalized"]],
        selected_base["Base"],
        selected_base["Quote"],
    )
    reference_live = fetch_live_cross_leg(
        CCXT_EXCHANGE_IDS[selected_reference["Exchange Normalized"]],
        selected_reference["Base"],
        selected_reference["Quote"],
    )

    base_last = safe_number(base_live.get("Last"))
    reference_last = safe_number(reference_live.get("Last"))
    implied_usdt = None
    if reference_last is not None and reference_last > 0:
        if selected_reference["Base"] == "USDT" and selected_reference["Quote"] in {"USD", "USDC"}:
            implied_usdt = reference_last
        elif selected_reference["Base"] == "USDC" and selected_reference["Quote"] == "USDT":
            implied_usdt = 1 / reference_last

    base_usdt_depth = safe_number(base_live.get("USDT Depth (2%)"))
    reference_usdt_depth = safe_number(reference_live.get("USDT Depth (2%)"))
    available_usdt_depth = (
        min(base_usdt_depth, reference_usdt_depth)
        if base_usdt_depth is not None and reference_usdt_depth is not None
        else None
    )
    observed_at = max(
        base_live.get("Observed At", pd.Timestamp.utcnow()),
        reference_live.get("Observed At", pd.Timestamp.utcnow()),
    )

    point = {
        "Time": observed_at,
        "Implied USDT Price (USD)": implied_usdt,
        "BASE/USDT Last": base_last,
        "BASE Leg USDT Depth (2%)": base_usdt_depth,
        "Reference Leg USDT Depth (2%)": reference_usdt_depth,
        "Available USDT Depth (2%)": available_usdt_depth,
    }
    history = st.session_state.setdefault("usdt_cross_history", [])
    if not history or history[-1]["Time"] != point["Time"]:
        history.append(point)
        del history[:-720]

    live_metrics = st.columns(4)
    live_metrics[0].metric("Implied USDT/USD", f"${implied_usdt:,.6f}" if implied_usdt else "—")
    live_metrics[1].metric("Peg Deviation", f"{(implied_usdt - 1) * 10_000:+,.2f} bps" if implied_usdt else "—")
    live_metrics[2].metric("BASE/USDT Last", f"{base_last:,.8f}" if base_last else "—")
    live_metrics[3].metric(
        "Available USDT Depth",
        f"{available_usdt_depth:,.0f} USDT" if available_usdt_depth is not None else "—",
    )

    if base_live.get("Status") != "OK":
        st.warning(f"BASE market: {base_live.get('Status')}")
    if reference_live.get("Status") != "OK":
        st.warning(f"Reference market: {reference_live.get('Status')}")

    history_df = pd.DataFrame(history)
    if not history_df.empty:
        price_chart = alt.Chart(history_df).mark_line(point=True).encode(
            x=alt.X("Time:T", title="Time (UTC)"),
            y=alt.Y("Implied USDT Price (USD):Q", title="USDT Price (USD)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Time:T", title="Time"),
                alt.Tooltip("Implied USDT Price (USD):Q", title="USDT/USD", format=".6f"),
            ],
        ).properties(title="Live Implied USDT Price", height=320)
        peg_rule = alt.Chart(pd.DataFrame({"peg": [1.0]})).mark_rule(strokeDash=[5, 5]).encode(y="peg:Q")
        st.altair_chart((price_chart + peg_rule).interactive(), width="stretch")

        depth_columns = [
            "BASE Leg USDT Depth (2%)",
            "Reference Leg USDT Depth (2%)",
            "Available USDT Depth (2%)",
        ]
        depth_long = history_df.melt(
            id_vars=["Time"], value_vars=depth_columns,
            var_name="Depth Series", value_name="Depth (USDT)",
        )
        depth_chart = alt.Chart(depth_long).mark_line(point=True).encode(
            x=alt.X("Time:T", title="Time (UTC)"),
            y=alt.Y("Depth (USDT):Q", title="USDT Depth"),
            color=alt.Color("Depth Series:N", title="Series"),
            tooltip=[
                alt.Tooltip("Time:T", title="Time"),
                alt.Tooltip("Depth Series:N", title="Series"),
                alt.Tooltip("Depth (USDT):Q", title="USDT Depth", format=",.0f"),
            ],
        ).properties(title="Live USDT-Side ±2% Depth", height=340).interactive()
        st.altair_chart(depth_chart, width="stretch")

    # Start the countdown only after the initial CoinGecko universe, summary
    # prices, selected live legs, and charts have all rendered. Placing this
    # near the top of the page can interrupt a slow first run when the
    # interval is short and keep the app trapped in its loading spinners.
    if live_auto_refresh and st_autorefresh is not None:
        st_autorefresh(
            interval=refresh_interval_seconds * 1_000,
            key="usdt_live_cross_refresh",
        )
