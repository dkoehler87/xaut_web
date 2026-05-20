# -*- coding: utf-8 -*-
"""
Streamlit live dashboard: COMEX front-month gold future vs XAUT/USDT across venues.

Feeds:
    - COMEX front-month gold future
    - Bitget XAUT/USDT via native public websocket
    - Bitfinex XAUT/USDT via native public websocket
    - Gate XAUT/USDT via native public websocket
    - OKX XAUT/USDT via native public websocket
    - SOFR via FRED CSV, used to infer a simple COMEX-implied spot price


Install:
    pip install streamlit pandas plotly tastytrade websockets requests

Important:
    The COMEX-implied spot calculation uses a simplified cost-of-carry model:
        spot = future_mid / exp(SOFR * years_to_expiry)
    It ignores storage, lease rates, convenience yield, and delivery frictions.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import websockets

from tastytrade import Session
from tastytrade.streamer import DXLinkStreamer
from tastytrade.dxfeed import Quote


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_COMEX_SYMBOL = "/GCM26:XCEC"
DEFAULT_MAX_POINTS = 1800
DEFAULT_INTERVAL_SEC = 3.0
DEFAULT_REFRESH_MS = 3000

BITGET_WS_URL = "wss://ws.bitget.com/v2/ws/public"
BITGET_INST_ID = "XAUTUSDT"

BITFINEX_WS_URL = "wss://api-pub.bitfinex.com/ws/2"
BITFINEX_SYMBOL = "tXAUT:UST"  # Bitfinex v2 websocket symbol for XAUt/USDt spot.

GATE_WS_URL = "wss://api.gateio.ws/ws/v4/"
GATE_PAIR = "XAUT_USDT"

OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_INST_ID = "XAUT-USDT"

FRED_SOFR_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"
MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# =============================================================================
# Thread-safe shared state
# =============================================================================

@dataclass
class VenueQuote:
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    last: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    ts: Optional[datetime] = None


@dataclass
class MarketState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    gc_symbol: str = DEFAULT_COMEX_SYMBOL
    bitget_inst_id: str = BITGET_INST_ID
    bitfinex_symbol: str = BITFINEX_SYMBOL
    bitfinex_symbols_tried: set = field(default_factory=set)
    gate_pair: str = GATE_PAIR
    okx_inst_id: str = OKX_INST_ID

    latest_gc_bid: Optional[float] = None
    latest_gc_ask: Optional[float] = None
    latest_gc_mid: Optional[float] = None
    latest_gc_time: Optional[datetime] = None

    sofr_rate_pct: Optional[float] = None
    sofr_rate_decimal: Optional[float] = None
    sofr_date: Optional[date] = None
    sofr_last_fetch: Optional[datetime] = None

    quotes: Dict[str, VenueQuote] = field(default_factory=lambda: {
        "Bitget": VenueQuote(),
        "Bitfinex": VenueQuote(),
        "Gate": VenueQuote(),
        "OKX": VenueQuote(),
    })

    rows: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_MAX_POINTS))
    logs: deque = field(default_factory=lambda: deque(maxlen=300))

    running: bool = False
    started_at: Optional[datetime] = None

    def log(self, msg: str) -> None:
        with self.lock:
            self.logs.appendleft(f"{datetime.now().strftime('%H:%M:%S')} | {msg}")


# =============================================================================
# Helpers
# =============================================================================

def d_to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def decimal_mid(bid, ask) -> Optional[float]:
    if bid is None or ask is None:
        return None
    try:
        return float((bid + ask) / Decimal("2"))
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def float_mid(bid, ask) -> Optional[float]:
    bid_f = safe_float(bid)
    ask_f = safe_float(ask)
    if bid_f is None or ask_f is None:
        return None
    return (bid_f + ask_f) / 2


def get_secret_any(*names: str, default: str = "") -> str:
    """Return first populated value from Streamlit secrets or environment variables."""
    for name in names:
        try:
            val = st.secrets.get(name)
            if val:
                return str(val)
        except Exception:
            pass

        val = os.getenv(name)
        if val:
            return val

    return default


def get_secret(name: str, default: str = "") -> str:
    # Backwards-compatible helper for any existing single-name callers.
    return get_secret_any(name, default=default)


def require_xaut_page_password() -> None:
    """Simple page-level password gate using XAUT_PASS from Streamlit secrets or env vars."""
    configured_password = get_secret_any("XAUT_PASS", default="")

    if not configured_password:
        st.error("XAUT_PASS is not configured in Streamlit secrets or environment variables.")
        st.stop()

    if st.session_state.get("xaut_vs_gold_authenticated"):
        return

    st.title("XAUT vs. Gold")
    entered_password = st.text_input("Password", type="password")

    if not entered_password:
        st.stop()

    if entered_password == configured_password:
        st.session_state["xaut_vs_gold_authenticated"] = True
        st.rerun()

    st.error("Incorrect password")
    st.stop()


def parse_gc_month_year(symbol: str) -> Tuple[int, int]:
    """Parse a TT-style future symbol like /GCM26:XCEC into month/year."""
    core = symbol.split(":", 1)[0].replace("/", "")
    # GC + month code + 2-digit year, e.g. GCM26
    month_code = core[2].upper()
    year_2 = int(core[3:5])
    month = MONTH_CODES[month_code]
    year = 2000 + year_2
    return month, year


def third_last_business_day(year: int, month: int) -> date:
    """Approximate GC last trade date: third-to-last business day of the delivery month."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)

    business_days = []
    while len(business_days) < 3:
        if d.weekday() < 5:
            business_days.append(d)
        d -= timedelta(days=1)
    return business_days[-1]


def gc_expiry_date(symbol: str) -> Optional[date]:
    """Approximate GC last trade date from the selected futures symbol."""
    try:
        month, year = parse_gc_month_year(symbol)
        return third_last_business_day(year, month)
    except Exception:
        return None


def gc_contract_month_name(symbol: str) -> str:
    """Return the COMEX contract month name, e.g. /GCM26:XCEC -> June."""
    try:
        month, _year = parse_gc_month_year(symbol)
        return MONTH_NAMES.get(month, "")
    except Exception:
        return ""


def days_to_expiry(symbol: str) -> Optional[int]:
    """Calendar days until approximate GC expiry / last trade date."""
    try:
        expiry = gc_expiry_date(symbol)
        if expiry is None:
            return None
        return max((expiry - datetime.now().date()).days, 0)
    except Exception:
        return None


def years_to_expiry(symbol: str) -> Optional[float]:
    try:
        expiry = gc_expiry_date(symbol)
        if expiry is None:
            return None
        days = (expiry - datetime.now().date()).days
        return max(days, 0) / 365.0
    except Exception:
        return None


def comex_implied_spot(future_mid: Optional[float], sofr_decimal: Optional[float], t_years: Optional[float]) -> Optional[float]:
    if future_mid is None or sofr_decimal is None or t_years is None:
        return None
    try:
        return future_mid / math.exp(sofr_decimal * t_years)
    except Exception:
        return None


def get_latest_sofr_from_fred() -> Tuple[Optional[date], Optional[float]]:
    """Fetch latest non-empty SOFR observation from FRED CSV. Returns date and rate as percent."""
    resp = requests.get(FRED_SOFR_CSV_URL, timeout=10)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    latest_date = None
    latest_value = None
    for row in reader:
        value = row.get("SOFR")
        obs_date = row.get("observation_date")
        if not value or value == "." or not obs_date:
            continue
        try:
            latest_date = datetime.strptime(obs_date, "%Y-%m-%d").date()
            latest_value = float(value)
        except Exception:
            continue

    return latest_date, latest_value


# =============================================================================
# Background async streams
# =============================================================================

async def stream_comex(state: MarketState, client_secret: str, refresh_token: str) -> None:
    """Stream COMEX quote updates"""
    try:
        session = Session(client_secret, refresh_token)
    except Exception as e:
        state.log(f"COMEX login error: {e}")
        return

    last_seen = None

    while state.running:
        try:
            async with DXLinkStreamer(session) as streamer:
                await streamer.subscribe(Quote, [state.gc_symbol])
                state.log(f"Subscribed COMEX {state.gc_symbol}")

                async for quote in streamer.listen(Quote):
                    if not state.running:
                        break

                    current = (quote.bid_price, quote.ask_price, quote.bid_size, quote.ask_size)
                    if current == last_seen:
                        continue
                    last_seen = current

                    bid = d_to_float(quote.bid_price)
                    ask = d_to_float(quote.ask_price)
                    mid = decimal_mid(quote.bid_price, quote.ask_price)

                    if mid is not None:
                        with state.lock:
                            state.latest_gc_bid = bid
                            state.latest_gc_ask = ask
                            state.latest_gc_mid = mid
                            state.latest_gc_time = datetime.now()

        except Exception as e:
            state.log(f"COMEX stream error: {e}; reconnecting in 5s")
            await asyncio.sleep(5)


async def refresh_sofr_loop(state: MarketState, refresh_seconds: int = 1800) -> None:
    while state.running:
        try:
            obs_date, rate_pct = await asyncio.to_thread(get_latest_sofr_from_fred)
            with state.lock:
                state.sofr_date = obs_date
                state.sofr_rate_pct = rate_pct
                state.sofr_rate_decimal = rate_pct / 100.0 if rate_pct is not None else None
                state.sofr_last_fetch = datetime.now()
            if rate_pct is not None:
                state.log(f"Fetched SOFR {rate_pct:.4f}% for {obs_date}")
        except Exception as e:
            state.log(f"SOFR fetch error: {e}")
        await asyncio.sleep(refresh_seconds)


async def bitget_ping_loop(ws, state: MarketState) -> None:
    while state.running:
        try:
            await asyncio.sleep(30)
            await ws.send("ping")
        except Exception:
            return


async def stream_bitget_xaut(state: MarketState) -> None:
    while state.running:
        ping_task = None
        try:
            async with websockets.connect(
                BITGET_WS_URL,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
                max_queue=1000,
            ) as ws:
                inst_id = state.bitget_inst_id.strip().upper()
                sub_msg = {
                    "op": "subscribe",
                    "args": [{"instType": "SPOT", "channel": "ticker", "instId": inst_id}],
                }
                await ws.send(json.dumps(sub_msg))
                state.log(f"Subscribed Bitget SPOT ticker {inst_id}")
                ping_task = asyncio.create_task(bitget_ping_loop(ws, state))

                async for raw_msg in ws:
                    if not state.running:
                        break
                    if raw_msg == "pong":
                        continue
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue
                    if "event" in msg:
                        if msg.get("event") == "error" or msg.get("code"):
                            state.log(f"Bitget event response: {msg}")
                        continue
                    data = msg.get("data") or []
                    if not data:
                        continue
                    tick = data[0]
                    bid = safe_float(tick.get("bidPr"))
                    ask = safe_float(tick.get("askPr"))
                    last = safe_float(tick.get("lastPr"))
                    bid_size = safe_float(tick.get("bidSz"))
                    ask_size = safe_float(tick.get("askSz"))
                    mid = float_mid(bid, ask)
                    ts_ms = safe_float(tick.get("ts"))
                    tick_time = datetime.fromtimestamp(ts_ms / 1000) if ts_ms else datetime.now()

                    if mid is not None:
                        with state.lock:
                            state.quotes["Bitget"] = VenueQuote(bid, ask, mid, last, bid_size, ask_size, tick_time)

        except Exception as e:
            state.log(f"Bitget websocket error: {e}; reconnecting in 5s")
            await asyncio.sleep(5)
        finally:
            if ping_task:
                ping_task.cancel()


def normalize_bitfinex_symbol(raw_symbol: str) -> str:
    """Return a Bitfinex v2 websocket trading symbol.

    Bitfinex v2 trading-pair websocket symbols must start with `t`.
    For pairs where the base/quote length is not the usual 3+3 format,
    Bitfinex uses a colon delimiter, e.g. `tXAUT:UST`.

    The UI displays XAUt/USDt as XAUT:UST; sending bare `XAUTUST` causes
    Bitfinex to strip the first character and interpret the pair as `AUTUST`,
    which is why the API returns code 10300.
    """
    s = (raw_symbol or "").strip().upper().replace("/", ":")
    if not s:
        return BITFINEX_SYMBOL

    # User may type XAUTUST because that is the compact exchange display format.
    # The websocket needs the type prefix and colon for this non-3-letter base.
    if s in {"XAUTUST", "XAUTUSDT", "XAUTUST", "XAUTUSDT"}:
        return "tXAUT:UST"

    if s in {"XAUT:UST", "XAUT:USDT", "XAUT/USD", "XAUT:USD"}:
        # Bitfinex's USDt ticker code is UST; XAUT/USD is separate and can be set manually as tXAUT:USD.
        return "tXAUT:UST" if "UST" in s or "USDT" in s else "tXAUT:USD"

    if s.startswith("T"):
        return "t" + s[1:]

    if ":" in s:
        return "t" + s

    return "t" + s


async def stream_bitfinex_xaut(state: MarketState) -> None:
    while state.running:
        try:
            async with websockets.connect(
                BITFINEX_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1000,
            ) as ws:
                user_symbol = state.bitfinex_symbol.strip()
                symbol = normalize_bitfinex_symbol(user_symbol)
                await ws.send(json.dumps({"event": "subscribe", "channel": "ticker", "symbol": symbol}))
                state.log(f"Subscribed Bitfinex ticker {symbol}")

                async for raw_msg in ws:
                    if not state.running:
                        break
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(msg, dict):
                        event = msg.get("event")
                        if event == "subscribed":
                            subscribed_symbol = msg.get("symbol") or symbol
                            state.log(f"Bitfinex subscription confirmed: {subscribed_symbol}")
                            continue
                        if event == "error":
                            state.log(f"Bitfinex error for {symbol}: {msg}")
                            # Do not tight-loop the same invalid symbol forever.
                            await asyncio.sleep(10)
                            break
                        continue

                    if not isinstance(msg, list) or len(msg) < 2:
                        continue
                    payload = msg[1]
                    if payload == "hb" or not isinstance(payload, list) or len(payload) < 10:
                        continue

                    # Ticker payload for trading pairs:
                    # [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_PERC, LAST_PRICE, VOLUME, HIGH, LOW]
                    bid = safe_float(payload[0])
                    bid_size = safe_float(payload[1])
                    ask = safe_float(payload[2])
                    ask_size = safe_float(payload[3])
                    last = safe_float(payload[6])
                    mid = float_mid(bid, ask)
                    if mid is not None:
                        with state.lock:
                            state.quotes["Bitfinex"] = VenueQuote(bid, ask, mid, last, bid_size, ask_size, datetime.now())

        except Exception as e:
            state.log(f"Bitfinex websocket error: {e}; reconnecting in 5s")
            await asyncio.sleep(5)


async def stream_gate_xaut(state: MarketState) -> None:
    while state.running:
        try:
            async with websockets.connect(
                GATE_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1000,
            ) as ws:
                pair = state.gate_pair.strip().upper()
                sub_msg = {
                    "time": int(time.time()),
                    "channel": "spot.tickers",
                    "event": "subscribe",
                    "payload": [pair],
                }
                await ws.send(json.dumps(sub_msg))
                state.log(f"Subscribed Gate spot.tickers {pair}")

                async for raw_msg in ws:
                    if not state.running:
                        break
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("event") == "subscribe":
                        if msg.get("result", {}).get("status") == "fail":
                            state.log(f"Gate subscription response: {msg}")
                        continue
                    if msg.get("channel") != "spot.tickers" or msg.get("event") != "update":
                        continue
                    result = msg.get("result") or {}
                    bid = safe_float(result.get("highest_bid"))
                    ask = safe_float(result.get("lowest_ask"))
                    last = safe_float(result.get("last"))
                    mid = float_mid(bid, ask)
                    ts = safe_float(msg.get("time_ms"))
                    tick_time = datetime.fromtimestamp(ts / 1000) if ts else datetime.now()
                    if mid is not None:
                        with state.lock:
                            state.quotes["Gate"] = VenueQuote(bid, ask, mid, last, None, None, tick_time)

        except Exception as e:
            state.log(f"Gate websocket error: {e}; reconnecting in 5s")
            await asyncio.sleep(5)


async def stream_okx_xaut(state: MarketState) -> None:
    while state.running:
        try:
            async with websockets.connect(
                OKX_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1000,
            ) as ws:
                inst_id = state.okx_inst_id.strip().upper()
                sub_msg = {
                    "op": "subscribe",
                    "args": [{"channel": "tickers", "instId": inst_id}],
                }
                await ws.send(json.dumps(sub_msg))
                state.log(f"Subscribed OKX tickers {inst_id}")

                async for raw_msg in ws:
                    if not state.running:
                        break
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("event") == "subscribe":
                        if msg.get("code") not in {None, "0"}:
                            state.log(f"OKX subscription response: {msg}")
                        continue
                    if msg.get("event") == "error":
                        state.log(f"OKX error: {msg}")
                        await asyncio.sleep(10)
                        break

                    data = msg.get("data") or []
                    if not data:
                        continue
                    tick = data[0]
                    bid = safe_float(tick.get("bidPx"))
                    ask = safe_float(tick.get("askPx"))
                    last = safe_float(tick.get("last"))
                    bid_size = safe_float(tick.get("bidSz"))
                    ask_size = safe_float(tick.get("askSz"))
                    mid = float_mid(bid, ask)
                    ts_ms = safe_float(tick.get("ts"))
                    tick_time = datetime.fromtimestamp(ts_ms / 1000) if ts_ms else datetime.now()
                    if mid is not None:
                        with state.lock:
                            state.quotes["OKX"] = VenueQuote(bid, ask, mid, last, bid_size, ask_size, tick_time)

        except Exception as e:
            state.log(f"OKX websocket error: {e}; reconnecting in 5s")
            await asyncio.sleep(5)


async def aggregate_rows(state: MarketState, interval_sec: float) -> None:
    while state.running:
        with state.lock:
            gc_mid = state.latest_gc_mid
            gc_bid = state.latest_gc_bid
            gc_ask = state.latest_gc_ask
            sofr_rate_decimal = state.sofr_rate_decimal
            sofr_rate_pct = state.sofr_rate_pct
            sofr_date = state.sofr_date
            t_years = years_to_expiry(state.gc_symbol)
            spot = comex_implied_spot(gc_mid, sofr_rate_decimal, t_years)
            quotes = {k: v for k, v in state.quotes.items()}

            if gc_mid is not None:
                row = {
                    "time": datetime.now(),
                    "gc_mid": gc_mid,
                    "gc_bid": gc_bid,
                    "gc_ask": gc_ask,
                    "sofr_pct": sofr_rate_pct,
                    "sofr_date": sofr_date,
                    "years_to_expiry": t_years,
                    "comex_spot": spot,
                }

                for venue, q in quotes.items():
                    prefix = venue.lower()
                    row[f"{prefix}_mid"] = q.mid
                    row[f"{prefix}_bid"] = q.bid
                    row[f"{prefix}_ask"] = q.ask
                    row[f"{prefix}_last"] = q.last
                    row[f"{prefix}_premium_vs_future_usd"] = q.mid - gc_mid if q.mid is not None else None
                    row[f"{prefix}_premium_vs_future_bps"] = ((q.mid - gc_mid) / gc_mid) * 10000 if q.mid is not None and gc_mid else None
                    row[f"{prefix}_premium_vs_spot_usd"] = q.mid - spot if q.mid is not None and spot is not None else None
                    row[f"{prefix}_premium_vs_spot_bps"] = ((q.mid - spot) / spot) * 10000 if q.mid is not None and spot else None

                state.rows.append(row)

        await asyncio.sleep(interval_sec)


async def run_streams(state: MarketState, client_secret: str, refresh_token: str, interval_sec: float) -> None:
    await asyncio.gather(
        stream_comex(state, client_secret, refresh_token),
        stream_bitget_xaut(state),
        stream_bitfinex_xaut(state),
        stream_gate_xaut(state),
        stream_okx_xaut(state),
        refresh_sofr_loop(state),
        aggregate_rows(state, interval_sec),
    )


def background_runner(state: MarketState, client_secret: str, refresh_token: str, interval_sec: float) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_streams(state, client_secret, refresh_token, interval_sec))
    except Exception as e:
        state.log(f"Background runner crashed: {e}")
    finally:
        try:
            loop.close()
        except Exception:
            pass


# =============================================================================
# Streamlit resource/session setup
# =============================================================================

@st.cache_resource
def get_state() -> MarketState:
    return MarketState()


def start_streaming(state: MarketState, client_secret: str, refresh_token: str, interval_sec: float) -> None:
    if state.running:
        return
    if not client_secret or not refresh_token:
        state.log("Missing TT credentials. Set TT_SECRET and TT_REFRESH as environment variables or Streamlit secrets.")
        return

    state.running = True
    state.started_at = datetime.now()

    t = threading.Thread(
        target=background_runner,
        args=(state, client_secret, refresh_token, interval_sec),
        daemon=True,
    )
    t.start()
    state.log("Started background streams")


def stop_streaming(state: MarketState) -> None:
    state.running = False
    state.log("Stop requested")


def configure_state(
    state: MarketState,
    gc_symbol: str,
    bitget_inst_id: str,
    bitfinex_symbol: str,
    gate_pair: str,
    okx_inst_id: str,
    max_points: int,
) -> None:
    """Update the shared engine configuration.

    These settings are process-global because the websocket engine is shared by
    every Streamlit user session through st.cache_resource.
    """
    with state.lock:
        state.gc_symbol = gc_symbol.strip()
        state.bitget_inst_id = bitget_inst_id.strip().upper()
        state.bitfinex_symbol = bitfinex_symbol.strip()
        state.gate_pair = gate_pair.strip().upper()
        state.okx_inst_id = okx_inst_id.strip().upper()
        state.rows = deque(state.rows, maxlen=int(max_points))


def restart_streaming(
    state: MarketState,
    client_secret: str,
    refresh_token: str,
    interval_sec: float,
) -> None:
    """Restart the single shared background engine.

    Use this after changing the COMEX contract or exchange symbols. The old
    daemon thread exits when it sees state.running=False; the new one starts
    after a short pause.
    """
    if state.running:
        stop_streaming(state)
        time.sleep(1.0)
    start_streaming(state, client_secret, refresh_token, interval_sec)


# =============================================================================
# UI
# =============================================================================

require_xaut_page_password()

st.title("COMEX Gold Futures vs XAUT/USDT Venues")
st.caption(
    "Live comparison using TT DXLink for COMEX, native exchange websockets for XAUT/USDT, "
    "and FRED SOFR for a simplified COMEX-implied spot estimate."
)

state = get_state()

client_secret = get_secret_any("TT_SECRET", "TT_CLIENT_SECRET", default="")
refresh_token = get_secret_any("TT_REFRESH", "TT_REFRESH_TOKEN", default="")

with st.sidebar:
    st.header("XAUT vs. Gold Settings")
    st.caption("One shared market-data engine is cached per running Streamlit process. All users read from the same websocket collector.")

    gc_symbol = st.text_input("COMEX symbol", value=state.gc_symbol)
    bitget_inst_id = st.text_input("Bitget instId", value=state.bitget_inst_id, help="Bitget spot symbols are compact, e.g. XAUTUSDT.")
    bitfinex_symbol = st.text_input("Bitfinex symbol", value=state.bitfinex_symbol, help="Default is tXAUT:UST. Bitfinex requires the leading t prefix and a colon for XAUT/USDt.")
    gate_pair = st.text_input("Gate pair", value=state.gate_pair, help="Gate spot pairs use underscore format, e.g. XAUT_USDT.")
    okx_inst_id = st.text_input("OKX instId", value=state.okx_inst_id, help="OKX spot symbols use dash format, e.g. XAUT-USDT.")

    max_points = st.number_input("Max stored points", min_value=100, max_value=10000, value=DEFAULT_MAX_POINTS, step=100)
    interval_sec = st.number_input("Aggregation interval seconds", min_value=0.5, max_value=30.0, value=DEFAULT_INTERVAL_SEC, step=0.5)
    refresh_ms = st.number_input("Dashboard refresh ms", min_value=1000, max_value=30000, value=DEFAULT_REFRESH_MS, step=1000)

    apply_restart = st.button("Apply settings & restart shared engine", width="stretch")

    st.divider()
    st.caption(f"Bitget WS: `{BITGET_WS_URL}`")
    st.caption(f"Bitfinex WS: `{BITFINEX_WS_URL}`")
    st.caption(f"Gate WS: `{GATE_WS_URL}`")
    st.caption(f"OKX WS: `{OKX_WS_URL}`")
    st.caption(f"SOFR CSV: `{FRED_SOFR_CSV_URL}`")

if apply_restart:
    configure_state(state, gc_symbol, bitget_inst_id, bitfinex_symbol, gate_pair, okx_inst_id, int(max_points))
    restart_streaming(state, client_secret, refresh_token, float(interval_sec))
else:
    # First user to open this page starts the single shared collector. Later
    # users hit this same function, but start_streaming is a no-op while running.
    configure_state(state, state.gc_symbol, state.bitget_inst_id, state.bitfinex_symbol, state.gate_pair, state.okx_inst_id, int(max_points))
    start_streaming(state, client_secret, refresh_token, float(interval_sec))


# =============================================================================
# Render dashboard
# =============================================================================

with state.lock:
    rows = list(state.rows)
    logs = list(state.logs)
    latest_quotes = {k: v for k, v in state.quotes.items()}
    latest = {
        "gc_bid": state.latest_gc_bid,
        "gc_ask": state.latest_gc_ask,
        "gc_mid": state.latest_gc_mid,
        "gc_time": state.latest_gc_time,
        "running": state.running,
        "started_at": state.started_at,
        "sofr_rate_pct": state.sofr_rate_pct,
        "sofr_date": state.sofr_date,
        "sofr_last_fetch": state.sofr_last_fetch,
        "t_years": years_to_expiry(state.gc_symbol),
        "days_to_expiry": days_to_expiry(state.gc_symbol),
        "contract_month": gc_contract_month_name(state.gc_symbol),
        "expiry_date": gc_expiry_date(state.gc_symbol),
    }

latest["comex_spot"] = comex_implied_spot(
    latest["gc_mid"],
    latest["sofr_rate_pct"] / 100.0 if latest["sofr_rate_pct"] is not None else None,
    latest["t_years"],
)

df = pd.DataFrame(rows)

status = "RUNNING" if latest["running"] else "STOPPED"
st.subheader(f"Status: {status}")

top_cols = st.columns([1, 1, 2])
with top_cols[0]:
    contract_month = latest.get("contract_month") or ""
    gc_mid_label = f"COMEX GC {contract_month} Mid" if contract_month else "COMEX GC Contract Mid"
    st.metric(gc_mid_label, f"{latest['gc_mid']:,.2f}" if latest["gc_mid"] is not None else "—")
with top_cols[1]:
    st.metric("COMEX Implied Spot Price", f"{latest['comex_spot']:,.2f}" if latest["comex_spot"] is not None else "—")

st.markdown("**Exchange XAUT/USDT mids**")
venue_cols = st.columns([1, 1, 1, 1])
for col, venue in zip(venue_cols, ["Bitget", "Bitfinex", "Gate", "OKX"]):
    q = latest_quotes.get(venue, VenueQuote())
    with col:
        st.metric(venue, f"{q.mid:,.4f}" if q.mid is not None else "—")
        if q.mid is not None and latest["comex_spot"] is not None:
            prem_bps = ((q.mid - latest["comex_spot"]) / latest["comex_spot"]) * 10000
            st.caption(f"vs implied spot: {prem_bps:,.2f} bps")
        elif q.mid is not None and latest["gc_mid"] is not None:
            prem_bps = ((q.mid - latest["gc_mid"]) / latest["gc_mid"]) * 10000
            st.caption(f"vs GC future: {prem_bps:,.2f} bps")
        else:
            st.caption("premium: —")

carry_cols = st.columns([1, 1, 1, 1])
with carry_cols[0]:
    st.metric("SOFR", f"{latest['sofr_rate_pct']:.4f}%" if latest["sofr_rate_pct"] is not None else "—")
with carry_cols[1]:
    st.metric("Days to Expiry", f"{latest['days_to_expiry']:,}" if latest.get("days_to_expiry") is not None else "—")
with carry_cols[2]:
    st.metric("Approx. Expiry", latest["expiry_date"].isoformat() if latest.get("expiry_date") else "—")

st.caption(
    f"COMEX bid/ask: "
    f"{latest['gc_bid']:,.2f} / {latest['gc_ask']:,.2f}" if latest["gc_bid"] is not None and latest["gc_ask"] is not None else "COMEX bid/ask: —"
)

quote_rows = []
for venue, q in latest_quotes.items():
    quote_rows.append({
        "Venue": venue,
        "Bid": q.bid,
        "Ask": q.ask,
        "Mid": q.mid,
        "Last": q.last,
        "Bid Size": q.bid_size,
        "Ask Size": q.ask_size,
        "Last Update": q.ts.strftime("%H:%M:%S") if q.ts else None,
        "Premium vs Spot (bps)": ((q.mid - latest["comex_spot"]) / latest["comex_spot"]) * 10000 if q.mid is not None and latest["comex_spot"] else None,
        "Premium vs Future (bps)": ((q.mid - latest["gc_mid"]) / latest["gc_mid"]) * 10000 if q.mid is not None and latest["gc_mid"] else None,
    })

st.dataframe(pd.DataFrame(quote_rows), width="stretch", hide_index=True)

if latest["gc_time"] or any(q.ts for q in latest_quotes.values()):
    venue_times = " | ".join(f"{v}: {q.ts.strftime('%H:%M:%S') if q.ts else '—'}" for v, q in latest_quotes.items())
    st.caption(
        f"Last COMEX update: {latest['gc_time'].strftime('%H:%M:%S') if latest['gc_time'] else '—'} | "
        f"{venue_times} | SOFR date: {latest['sofr_date'] or '—'}"
    )

if df.empty:
    st.info("Waiting for the shared market-data engine to receive COMEX and XAUT venue quotes")
else:
    df = df.sort_values("time")

    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(x=df["time"], y=df["gc_mid"], mode="lines", name="COMEX GC Future"))
    if "comex_spot" in df.columns:
        fig_prices.add_trace(go.Scatter(x=df["time"], y=df["comex_spot"], mode="lines", name="COMEX Implied Spot"))
    for venue in ["bitget", "bitfinex", "gate", "okx"]:
        col = f"{venue}_mid"
        if col in df.columns and df[col].notna().any():
            fig_prices.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=f"{venue.title()} XAUT"))
    fig_prices.update_layout(
        title="Live Absolute Prices",
        xaxis_title="Time",
        yaxis_title="USD / oz",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_prices, width="stretch")

    fig_prem = go.Figure()
    for venue in ["bitget", "bitfinex", "gate", "okx"]:
        col = f"{venue}_premium_vs_spot_bps"
        fallback_col = f"{venue}_premium_vs_future_bps"
        if col in df.columns and df[col].notna().any():
            fig_prem.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=f"{venue.title()} vs COMEX Spot"))
        elif fallback_col in df.columns and df[fallback_col].notna().any():
            fig_prem.add_trace(go.Scatter(x=df["time"], y=df[fallback_col], mode="lines", name=f"{venue.title()} vs GC Future"))
    fig_prem.add_hline(y=0, line_dash="dash")
    fig_prem.update_layout(
        title="XAUT Premium / Discount",
        xaxis_title="Time",
        yaxis_title="bps",
        height=390,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_prem, width="stretch")

    # Normalized chart
    fig_norm = go.Figure()
    if df["gc_mid"].notna().any():
        fig_norm.add_trace(go.Scatter(x=df["time"], y=df["gc_mid"] / df["gc_mid"].dropna().iloc[0] * 100, mode="lines", name="COMEX GC normalized"))
    if "comex_spot" in df.columns and df["comex_spot"].notna().any():
        fig_norm.add_trace(go.Scatter(x=df["time"], y=df["comex_spot"] / df["comex_spot"].dropna().iloc[0] * 100, mode="lines", name="COMEX Spot normalized"))
    for venue in ["bitget", "bitfinex", "gate", "okx"]:
        col = f"{venue}_mid"
        if col in df.columns and df[col].notna().any():
            base = df[col].dropna().iloc[0]
            fig_norm.add_trace(go.Scatter(x=df["time"], y=df[col] / base * 100, mode="lines", name=f"{venue.title()} XAUT normalized"))
    fig_norm.update_layout(
        title="Normalized Live Price Comparison",
        xaxis_title="Time",
        yaxis_title="Start = 100",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_norm, width="stretch")

    with st.expander("Latest raw points"):
        st.dataframe(df.tail(25).sort_values("time", ascending=False), width="stretch", hide_index=True)

with st.expander("Logs", expanded=False):
    if logs:
        st.code("\n".join(logs[:120]))
    else:
        st.write("No logs yet.")

st.caption(
    "COMEX Implied Spot Price is a simplified carry-adjusted estimate: future_mid / exp(SOFR × years_to_expiry). "
    "It ignores storage costs, gold lease rates, convenience yield, exchange delivery mechanics, and holidays."
)

# Auto-refresh. This keeps the app live without adding streamlit-autorefresh.
time.sleep(float(refresh_ms) / 1000.0)
st.rerun()
