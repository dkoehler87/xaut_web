# -*- coding: utf-8 -*-
"""
pages/market_lookup.py
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Market Lookup", layout="wide")
st.title("Market Lookup")
st.caption("Discover CCXT markets, filter symbols, then run configurable liquidity-band snapshots.")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def clean_ccxt_error(exc: Exception) -> str:
    msg = str(exc)
    return msg[:700] + ("..." if len(msg) > 700 else "")


def make_exchange(exchange_id: str, timeout_ms: int = 10_000) -> ccxt.Exchange:
    if not exchange_id:
        raise ValueError("Please select an exchange.")

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange '{exchange_id}' not found in ccxt.")

    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class(
        {
            "enableRateLimit": True,
            "timeout": int(timeout_ms),
        }
    )

def drop_constant_columns(
    df: pd.DataFrame,
    min_rows: int = 2,
) -> pd.DataFrame:
    # Don't drop constant columns when we have
    # fewer than min_rows rows
    if len(df) < min_rows:
        return df

    keep_cols = []

    always_keep = {
        "symbol",
        "base",
        "quote",
        "type",
        "bid",
        "ask",
        "mid",
        "last",
        "spread_bps",
        "error",
    }

    for col in df.columns:
        if col in always_keep:
            keep_cols.append(col)
            continue

        unique_values = df[col].dropna().unique()

        if len(unique_values) > 1:
            keep_cols.append(col)

    return df[keep_cols]

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Keep the expected market-price schema visible and exportable even when
    # a particular exchange's bulk ticker endpoint omits some BBO fields.
    always_keep = {
        "symbol",
        "base",
        "quote",
        "type",
        "last",
        "bid",
        "ask",
        "mid",
        "spread_bps",
    }

    cols_to_keep = []

    for col in df.columns:
        if col in always_keep:
            cols_to_keep.append(col)
            continue

        series = df[col]

        # Remove columns where every value is null/blank
        if series.notna().any():
            non_null = series.dropna()

            if len(non_null) > 0:
                if not (
                    non_null.astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .isna()
                    .all()
                ):
                    cols_to_keep.append(col)

    return df[cols_to_keep]


def market_type_label(market: Dict[str, Any]) -> str:
    if market.get("spot"):
        return "spot"
    if market.get("swap"):
        return "swap"
    if market.get("future"):
        return "future"
    if market.get("option"):
        return "option"
    if market.get("margin"):
        return "margin"
    return market.get("type") or "unknown"


def calc_mid(bid: Any, ask: Any) -> Optional[float]:
    bid_f = safe_float(bid)
    ask_f = safe_float(ask)
    if bid_f is None or ask_f is None or bid_f <= 0 or ask_f <= 0:
        return None
    return (bid_f + ask_f) / 2.0


def calc_spread_bps(bid: Any, ask: Any) -> Optional[float]:
    bid_f = safe_float(bid)
    ask_f = safe_float(ask)
    mid = calc_mid(bid_f, ask_f)
    if bid_f is None or ask_f is None or mid is None or mid <= 0:
        return None
    return (ask_f - bid_f) / mid * 10_000


def quote_depth_within_bps(
    side_levels: List[List[float]],
    mid: Optional[float],
    side: str,
    bps: float,
) -> float:
    if mid is None or mid <= 0:
        return 0.0

    total_quote = 0.0

    if side == "bid":
        cutoff = mid * (1 - bps / 10_000.0)
        for level in side_levels:
            if len(level) < 2:
                continue
            price = safe_float(level[0])
            amount = safe_float(level[1])
            if price is not None and amount is not None and price >= cutoff:
                total_quote += price * amount

    elif side == "ask":
        cutoff = mid * (1 + bps / 10_000.0)
        for level in side_levels:
            if len(level) < 2:
                continue
            price = safe_float(level[0])
            amount = safe_float(level[1])
            if price is not None and amount is not None and price <= cutoff:
                total_quote += price * amount

    return float(total_quote)

def checkbox_aligned(label: str, value: bool = False, key: str | None = None) -> bool:
    st.write("")
    st.write("")
    return st.checkbox(label, value=value, key=key)


def parse_bps_bands(raw: str) -> List[float]:
    bands = []

    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue

        try:
            value = float(part)
            if 0 < value <= 10_000:
                bands.append(value)
        except ValueError:
            pass

    seen = set()
    out = []

    for band in bands:
        if band not in seen:
            seen.add(band)
            out.append(band)

    return out[:20]


def safe_fetch_tickers(
    exchange: ccxt.Exchange,
    symbols: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not exchange.has.get("fetchTickers"):
        return {}, "fetchTickers not supported by this exchange"

    try:
        if symbols:
            try:
                return exchange.fetch_tickers(symbols), None
            except Exception:
                # Some CCXT exchanges expose fetchTickers but do not accept a
                # symbols argument. Retry their all-tickers form.
                return exchange.fetch_tickers(), None
        return exchange.fetch_tickers(), None
    except Exception as exc:
        return {}, f"fetchTickers failed: {clean_ccxt_error(exc)}"


def safe_fetch_ticker(exchange: ccxt.Exchange, symbol: str) -> Tuple[Dict[str, Any], Optional[str]]:
    if not exchange.has.get("fetchTicker"):
        return {}, "fetchTicker not supported by this exchange"

    try:
        return exchange.fetch_ticker(symbol), None
    except Exception as exc:
        return {}, f"fetchTicker failed for {symbol}: {clean_ccxt_error(exc)}"


def dataframe_to_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str,
    comma_format_numeric: bool = False,
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}},
    ) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        header_format = workbook.add_format(
            {
                "bold": True,
                "text_wrap": False,
                "valign": "top",
                "border": 1,
            }
        )

        number_format = workbook.add_format({"num_format": "#,##0.00"})
        integer_format = workbook.add_format({"num_format": "#,##0"})
        default_numeric_format = workbook.add_format({"num_format": "0.########"})

        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)

            series = df[col_name]
            sample = series.dropna().astype(str).head(100)
            max_content_len = max([len(str(col_name))] + [len(x) for x in sample]) if len(sample) else len(str(col_name))
            width = min(max(max_content_len + 2, 10), 32)

            if pd.api.types.is_numeric_dtype(series):
                if comma_format_numeric:
                    if any(key in col_name.lower() for key in ["num_", "count", "nonce"]):
                        worksheet.set_column(col_num, col_num, width, integer_format)
                    else:
                        worksheet.set_column(col_num, col_num, width, number_format)
                else:
                    worksheet.set_column(col_num, col_num, width, default_numeric_format)
            else:
                worksheet.set_column(col_num, col_num, width)

        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))

    return output.getvalue()


def excel_download_button(
    df: pd.DataFrame,
    label: str,
    filename: str,
    sheet_name: str,
    key: str,
    comma_format_numeric: bool = False,
) -> None:
    if df is None or df.empty:
        return

    excel_bytes = dataframe_to_excel_bytes(
        df=df,
        sheet_name=sheet_name,
        comma_format_numeric=comma_format_numeric,
    )

    st.download_button(
        label=label,
        data=excel_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_markets_df(
    exchange_id: str,
    active_only: bool,
    market_type_filter: str,
    include_tickers: bool,
    ticker_mode: str,
    max_per_symbol_tickers: int,
    timeout_ms: int,
) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []
    exchange = make_exchange(exchange_id, timeout_ms=timeout_ms)

    try:
        markets = exchange.load_markets()
    except Exception as exc:
        raise RuntimeError(f"load_markets failed for {exchange_id}: {clean_ccxt_error(exc)}") from exc

    ticker_by_symbol = {}

    symbols_for_tickers = []
    for symbol, market in markets.items():
        market_type = market_type_label(market)
        if market_type_filter != "All" and market_type != market_type_filter:
            continue
        if active_only and market.get("active") is not True:
            continue
        symbols_for_tickers.append(symbol)

    if include_tickers:
        if ticker_mode == "Auto":
            if exchange_id.lower() in {"bitso"}:
                warnings.append(
                    "Skipped ticker enrichment in Auto mode for Bitso to avoid slow/hanging fetchTickers calls."
                )
            elif exchange.has.get("fetchTickers"):
                ticker_by_symbol, err = safe_fetch_tickers(exchange, symbols_for_tickers)
                if err:
                    warnings.append(err)
            else:
                warnings.append(
                    "Bulk ticker retrieval is unsupported; Auto mode skipped price enrichment. "
                    "No per-symbol ticker requests were made."
                )

        elif ticker_mode == "Bulk fetchTickers":
            ticker_by_symbol, err = safe_fetch_tickers(exchange)
            if err:
                warnings.append(err)

        elif ticker_mode == "Per-symbol fetchTicker":
            if not exchange.has.get("fetchTicker"):
                warnings.append("fetchTicker not supported by this exchange.")
            else:
                limited_symbols = symbols_for_tickers[: max(0, int(max_per_symbol_tickers))]
                if len(symbols_for_tickers) > len(limited_symbols):
                    warnings.append(
                        f"Per-symbol ticker enrichment limited to first {len(limited_symbols)} symbols "
                        f"out of {len(symbols_for_tickers)}."
                    )

                progress = st.progress(0, text="Fetching per-symbol tickers...")
                for i, symbol in enumerate(limited_symbols, start=1):
                    ticker, err = safe_fetch_ticker(exchange, symbol)
                    if ticker:
                        ticker_by_symbol[symbol] = ticker
                    if err and len(warnings) < 8:
                        warnings.append(err)
                    progress.progress(
                        i / max(len(limited_symbols), 1),
                        text=f"Fetching ticker {i}/{len(limited_symbols)}",
                    )
                progress.empty()

    rows = []

    for symbol, market in markets.items():
        market_type = market_type_label(market)

        if market_type_filter != "All" and market_type != market_type_filter:
            continue

        if active_only and market.get("active") is not True:
            continue

        ticker = ticker_by_symbol.get(symbol, {}) or {}
        bid = safe_float(ticker.get("bid"))
        ask = safe_float(ticker.get("ask"))

        rows.append(
            {
                "symbol": market.get("symbol") or symbol,
                "id": market.get("id"),
                "type": market_type,
                "base": market.get("base"),
                "quote": market.get("quote"),
                "settle": market.get("settle"),
                "active": market.get("active"),
                "bid": bid,
                "ask": ask,
                "mid": calc_mid(bid, ask),
                "last": safe_float(ticker.get("last")),
                "spread_bps": calc_spread_bps(bid, ask),
                "contract": market.get("contract"),
                "linear": market.get("linear"),
                "inverse": market.get("inverse"),
                "maker_fee": market.get("maker"),
                "taker_fee": market.get("taker"),
                "price_precision": (market.get("precision") or {}).get("price"),
                "amount_precision": (market.get("precision") or {}).get("amount"),
                "min_size": ((market.get("limits") or {}).get("amount") or {}).get("min"),
                "max_size": ((market.get("limits") or {}).get("amount") or {}).get("max"),
                "min_price": ((market.get("limits") or {}).get("price") or {}).get("min"),
                "max_price": ((market.get("limits") or {}).get("price") or {}).get("max"),
                "min_notional": ((market.get("limits") or {}).get("cost") or {}).get("min"),
                "max_notional": ((market.get("limits") or {}).get("cost") or {}).get("max"),
                "base_volume": ticker.get("baseVolume"),
                "quote_volume": ticker.get("quoteVolume"),
                "ticker_timestamp": ticker.get("timestamp"),
                "ticker_datetime": ticker.get("datetime"),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        sort_cols = [c for c in ["type", "quote", "base", "symbol"] if c in df.columns]
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return df, warnings


def apply_market_filters(
    df: pd.DataFrame,
    symbol_filter: str,
    base_filter: str,
    quote_filter: str,
    general_filter: str,
) -> pd.DataFrame:
    out = df.copy()

    def contains(col: str, val: str) -> pd.Series:
        if not val or col not in out.columns:
            return pd.Series([True] * len(out), index=out.index)
        return out[col].astype(str).str.contains(val, case=False, na=False, regex=False)

    out = out[contains("symbol", symbol_filter)]
    out = out[contains("base", base_filter)]
    out = out[contains("quote", quote_filter)]

    if general_filter:
        search_cols = [c for c in ["symbol", "id", "type", "base", "quote", "settle"] if c in out.columns]
        mask = pd.Series(False, index=out.index)
        for col in search_cols:
            mask = mask | out[col].astype(str).str.contains(general_filter, case=False, na=False, regex=False)
        out = out[mask]

    return out.reset_index(drop=True)


def format_liquidity_dataframe_for_display(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and any(
            key in col.lower()
            for key in [
                "bid",
                "ask",
                "mid",
                "spread",
                "depth",
                "quote",
                "amount",
                "precision",
                "num_",
                "timestamp",
                "nonce",
            ]
        )
    ]

    format_dict = {}

    for col in numeric_cols:
        lower = col.lower()
        if any(key in lower for key in ["num_", "timestamp", "nonce"]):
            format_dict[col] = "{:,.0f}"
        elif "spread_bps" in lower:
            format_dict[col] = "{:,.2f}"
        else:
            format_dict[col] = "{:,.2f}"

    return df.style.format(format_dict, na_rep="")


def get_liquidity_snapshot(
    exchange_id: str,
    symbols: List[str],
    bps_bands: List[float],
    limit: Optional[int],
    timeout_ms: int,
    spot_only: bool,
) -> pd.DataFrame:
    exchange = make_exchange(exchange_id, timeout_ms=timeout_ms)

    try:
        markets = exchange.load_markets()
    except Exception as exc:
        raise RuntimeError(f"load_markets failed for {exchange_id}: {clean_ccxt_error(exc)}") from exc

    rows = []

    for symbol in symbols:
        if symbol not in markets:
            rows.append({"symbol": symbol, "error": "symbol not found in exchange.load_markets()"})
            continue

        market = markets[symbol]

        if spot_only and not market.get("spot"):
            rows.append({"symbol": symbol, "error": "symbol exists but is not a spot market"})
            continue

        try:
            order_book = (
                exchange.fetch_order_book(symbol, int(limit))
                if limit is not None and int(limit) > 0
                else exchange.fetch_order_book(symbol)
            )
        except Exception as exc:
            rows.append({"symbol": symbol, "error": clean_ccxt_error(exc)})
            continue

        bids = order_book.get("bids", []) or []
        asks = order_book.get("asks", []) or []

        best_bid = safe_float(bids[0][0]) if bids and len(bids[0]) >= 2 else None
        best_bid_amt = safe_float(bids[0][1]) if bids and len(bids[0]) >= 2 else None
        best_ask = safe_float(asks[0][0]) if asks and len(asks[0]) >= 2 else None
        best_ask_amt = safe_float(asks[0][1]) if asks and len(asks[0]) >= 2 else None

        mid = calc_mid(best_bid, best_ask)

        row = {
            "symbol": symbol,
            "type": market_type_label(market),
            "base": market.get("base"),
            "quote": market.get("quote"),
            "active": market.get("active"),
            "bid": best_bid,
            "ask": best_ask,
            "mid": mid,
            "spread_bps": calc_spread_bps(best_bid, best_ask),
            "tob_bid_depth_quote": best_bid * best_bid_amt
            if best_bid is not None and best_bid_amt is not None
            else None,
            "tob_ask_depth_quote": best_ask * best_ask_amt
            if best_ask is not None and best_ask_amt is not None
            else None,
            "num_bids": len(bids),
            "num_asks": len(asks),
            "price_precision": (market.get("precision") or {}).get("price"),
            "amount_precision": (market.get("precision") or {}).get("amount"),
            "timestamp": order_book.get("timestamp"),
            "datetime": order_book.get("datetime"),
            "nonce": order_book.get("nonce"),
            "error": None,
        }

        for bps in bps_bands:
            label = str(int(bps)) if float(bps).is_integer() else str(bps).replace(".", "_")
            bid_depth = quote_depth_within_bps(bids, mid, "bid", bps)
            ask_depth = quote_depth_within_bps(asks, mid, "ask", bps)

            row[f"bid_depth_{label}bps_quote"] = bid_depth
            row[f"ask_depth_{label}bps_quote"] = ask_depth
            row[f"total_depth_{label}bps_quote"] = bid_depth + ask_depth

        rows.append(row)

        if getattr(exchange, "rateLimit", None):
            time.sleep(min(float(exchange.rateLimit) / 1000.0, 0.35))

    return pd.DataFrame(rows)


st.header("Exchange Market Discovery")

with st.expander("Market Discovery Settings", expanded=True):

    exchange_row = st.columns([2, 1, 1])

    with exchange_row[0]:
        exchange_id = st.selectbox(
            "Exchange",
            options=[""] + sorted(ccxt.exchanges),
            index=0,
            format_func=lambda x: "Select exchange..." if x == "" else x,
        )

    with exchange_row[1]:
        timeout_ms = st.number_input(
            "Timeout (ms)",
            min_value=1_000,
            max_value=120_000,
            value=10_000,
            step=1_000,
        )

    with exchange_row[2]:
        active_only = checkbox_aligned("Active only", value=False, key="active_only")

    settings_row = st.columns(4)

    with settings_row[0]:
        market_type_filter = st.selectbox(
            "Market Type",
            ["spot", "All", "swap", "future", "option", "margin"],
            index=0,
        )

    with settings_row[1]:
        include_tickers = checkbox_aligned(
            "Include Prices",
            value=True,
            key="include_tickers",
        )

    with settings_row[2]:
        ticker_mode = st.selectbox(
            "Ticker Mode",
            [
                "Auto",
                "Bulk fetchTickers",
                "Per-symbol fetchTicker",
            ],
            disabled=not include_tickers,
            help="Auto uses bulk tickers only. It never falls back to per-market requests.",
        )

    with settings_row[3]:
        max_per_symbol_tickers = st.number_input(
            "Max Tickers",
            min_value=1,
            max_value=2000,
            value=50,
            step=25,
            disabled=(
                not include_tickers
                or ticker_mode != "Per-symbol fetchTicker"
            ),
        )

    fetch_markets = st.button(
        "Pull Markets",
        type="primary",
        disabled=not exchange_id,
    )

if "markets_df" not in st.session_state:
    st.session_state["markets_df"] = pd.DataFrame()

if "markets_exchange_id" not in st.session_state:
    st.session_state["markets_exchange_id"] = None

if "market_warnings" not in st.session_state:
    st.session_state["market_warnings"] = []

if "liq_df" not in st.session_state:
    st.session_state["liq_df"] = pd.DataFrame()

if "liq_exchange_id" not in st.session_state:
    st.session_state["liq_exchange_id"] = None

if fetch_markets:
    with st.spinner(f"Loading markets for {exchange_id}..."):
        try:
            df_markets, warnings = get_markets_df(
                exchange_id=exchange_id,
                active_only=active_only,
                market_type_filter=market_type_filter,
                include_tickers=include_tickers,
                ticker_mode=ticker_mode,
                max_per_symbol_tickers=int(max_per_symbol_tickers),
                timeout_ms=int(timeout_ms),
            )

            st.session_state["markets_df"] = df_markets
            st.session_state["markets_exchange_id"] = exchange_id
            st.session_state["market_warnings"] = warnings

            st.session_state["liq_df"] = pd.DataFrame()
            st.session_state["liq_exchange_id"] = None

        except Exception as exc:
            st.error(clean_ccxt_error(exc))

markets_df = st.session_state.get("markets_df", pd.DataFrame())

if not markets_df.empty:
    st.subheader("Markets")

    for warning in st.session_state.get("market_warnings", []):
        st.warning(warning)

    filter_cols = st.columns([1, 1, 1, 2])

    with filter_cols[0]:
        symbol_filter = st.text_input("Filter symbol", placeholder="e.g. XAUT, BTC/USD")

    with filter_cols[1]:
        base_filter = st.text_input("Filter base", placeholder="e.g. BTC")

    with filter_cols[2]:
        quote_filter = st.text_input("Filter quote", placeholder="e.g. USD, USDT")

    with filter_cols[3]:
        general_filter = st.text_input(
            "General market search",
            placeholder="Search symbol, id, type, base, quote, settle",
        )

    filtered_markets_df = apply_market_filters(
        markets_df,
        symbol_filter=symbol_filter,
        base_filter=base_filter,
        quote_filter=quote_filter,
        general_filter=general_filter,
    )
    
    display_markets_df = drop_empty_columns(filtered_markets_df)
    display_markets_df = drop_constant_columns(display_markets_df)
    
    current_exchange_id = st.session_state.get("markets_exchange_id") or exchange_id

    st.caption(
        f"Showing {len(filtered_markets_df):,} of {len(markets_df):,} markets for {current_exchange_id}."
    )

    st.dataframe(
        display_markets_df,
        width="stretch",
        hide_index=True,
        height=430,
    )

    excel_download_button(
        display_markets_df,
        "Download Excel",
        f"{current_exchange_id}_markets.xlsx",
        "Markets",
        "download_markets_excel",
        comma_format_numeric=False,
    )

else:
    filtered_markets_df = pd.DataFrame()
    st.info("Choose an exchange and click **Pull Markets**.")


st.header("Liquidity Snapshot")
st.write("Select markets from the pulled/filtered universe, then fetch detailed order-book liquidity.")

if not markets_df.empty:
    liquidity_symbol_source = filtered_markets_df if not filtered_markets_df.empty else markets_df
    available_symbols = liquidity_symbol_source["symbol"].dropna().astype(str).tolist()
else:
    available_symbols = []

with st.expander("Liquidity inputs", expanded=True):
    selection_mode = st.radio(
        "Markets",
        ["Select", "All filtered", "Paste manually"],
        horizontal=True,
    )

    if selection_mode == "All filtered":
        selected_symbols = available_symbols
        st.caption(f"{len(selected_symbols):,} filtered market(s) selected.")

    elif selection_mode == "Paste manually":
        pasted_markets = st.text_area(
            "Paste markets",
            placeholder="BTC/USD\nETH/USD\nXAUT/USD",
            height=90,
        )

        selected_symbols = [
            symbol.strip()
            for symbol in pasted_markets.replace(",", "\n").splitlines()
            if symbol.strip()
        ]

    else:
        selected_symbols = st.multiselect(
            "Select markets",
            options=available_symbols,
            default=[],
        )

    liq_row = st.columns(3)

    with liq_row[0]:
        order_book_limit = st.number_input(
            "Order book limit",
            min_value=0,
            max_value=5_000,
            value=200,
            step=50,
        )

    with liq_row[1]:
        bps_bands_raw = st.text_input(
            "Bands, bps",
            value="50,100,200",
        )

    with liq_row[2]:
        spot_only = checkbox_aligned("Require spot", value=True, key="spot_only")

bps_bands = parse_bps_bands(bps_bands_raw)
MAX_LIQUIDITY_SYMBOLS = 100
if len(selected_symbols) > MAX_LIQUIDITY_SYMBOLS:
    st.warning(
        f"Liquidity snapshots are limited to {MAX_LIQUIDITY_SYMBOLS} markets per run; "
        "narrow the filters or selection."
    )

if not bps_bands:
    st.warning("Enter at least one valid positive bps band.")

run_liquidity = st.button(
    "Run Liquidity Snapshot",
    disabled=(
        not selected_symbols
        or not bps_bands
        or len(selected_symbols) > MAX_LIQUIDITY_SYMBOLS
    ),
)

if run_liquidity:
    liq_exchange_id = st.session_state.get("markets_exchange_id") or exchange_id

    with st.spinner(f"Fetching order books for {len(selected_symbols)} market(s) on {liq_exchange_id}..."):
        try:
            liq_df = get_liquidity_snapshot(
                exchange_id=liq_exchange_id,
                symbols=selected_symbols,
                bps_bands=bps_bands,
                limit=None if int(order_book_limit) == 0 else int(order_book_limit),
                timeout_ms=int(timeout_ms),
                spot_only=spot_only,
            )

            st.session_state["liq_df"] = liq_df
            st.session_state["liq_exchange_id"] = liq_exchange_id

        except Exception as exc:
            st.error(clean_ccxt_error(exc))

liq_df = st.session_state.get("liq_df", pd.DataFrame())

if not liq_df.empty:
    st.subheader("Liquidity Results")
    st.caption(f"Exchange: {st.session_state.get('liq_exchange_id', exchange_id)}")
    
    display_liq_df = drop_empty_columns(liq_df)
    display_liq_df = drop_constant_columns(display_liq_df)

    st.dataframe(
        format_liquidity_dataframe_for_display(display_liq_df),
        width="stretch",
        hide_index=True,
        height=430,
    )

    excel_download_button(
        display_liq_df,
        "Download Excel",
        f"{st.session_state.get('liq_exchange_id', exchange_id)}_liquidity_snapshot.xlsx",
        "Liquidity",
        "download_liquidity_excel",
        comma_format_numeric=True,
    )
