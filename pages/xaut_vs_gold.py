# -*- coding: utf-8 -*-
"""
Streamlit live dashboard

"""

from __future__ import annotations

import asyncio
import json
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

DEFAULT_COMEX_SYMBOL = "/GCQ26:XCEC"  # August 2026 GC contract.
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

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"
BYBIT_XAUT_SYMBOL = "XAUTUSDT"
BYBIT_USDC_SYMBOL = "USDCUSDT"

OANDA_DEFAULT_STREAM_URL = "https://stream-fxpractice.oanda.com"
OANDA_INSTRUMENT = "XAU_USD"

MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
VENUES = ["Bitget", "Bitfinex", "Gate", "OKX", "Bybit"]
VENUE_KEYS = ["bitget", "bitfinex", "gate", "okx", "bybit"]


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
    gate_pair: str = GATE_PAIR
    okx_inst_id: str = OKX_INST_ID
    bybit_xaut_symbol: str = BYBIT_XAUT_SYMBOL
    bybit_usdc_symbol: str = BYBIT_USDC_SYMBOL

    latest_gc_bid: Optional[float] = None
    latest_gc_ask: Optional[float] = None
    latest_gc_mid: Optional[float] = None
    latest_gc_time: Optional[datetime] = None

    latest_oanda_bid: Optional[float] = None
    latest_oanda_ask: Optional[float] = None
    latest_oanda_mid: Optional[float] = None
    latest_oanda_time: Optional[datetime] = None

    bybit_usdc_quote: VenueQuote = field(default_factory=VenueQuote)

    quotes: Dict[str, VenueQuote] = field(default_factory=lambda: {
        "Bitget": VenueQuote(),
        "Bitfinex": VenueQuote(),
        "Gate": VenueQuote(),
        "OKX": VenueQuote(),
        "Bybit": VenueQuote(),
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


def normalized_xaut_price(price: Optional[float], usdcusdt_mid: Optional[float]) -> Optional[float]:
    """Convert USDT-quoted XAUT into an approximate USD/USDC-normalized price.

    User requested multiplying XAUT/USDT by the inverse of Bybit USDC/USDT.
    """
    if price is None or usdcusdt_mid is None or usdcusdt_mid == 0:
        return None
    return price * (1.0 / usdcusdt_mid)


def premium_bps(price: Optional[float], reference: Optional[float]) -> Optional[float]:
    if price is None or reference is None or reference == 0:
        return None
    return ((price - reference) / reference) * 10000


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
    """Parse a TT-style future symbol like /GCQ26:XCEC into month/year."""
    core = symbol.split(":", 1)[0].replace("/", "")
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
    """Return the COMEX contract month name, e.g. /GCQ26:XCEC -> August."""
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


# =============================================================================
# Background streams
# =============================================================================

async def stream_comex(state: MarketState, client_secret: str, refresh_token: str) -> None:
    """Stream COMEX quote updates."""
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


def stream_oanda_spot_blocking(
    state: MarketState,
    api_key: str,
    account_id: str,
    stream_url: str,
    instrument: str = OANDA_INSTRUMENT,
) -> None:
    """Blocking OANDA streaming loop. Run inside asyncio.to_thread()."""
    if not api_key or not account_id:
        state.log("Missing OANDA credentials. Set OANDA_DEMO_API_KEY/OANDA_DEMO_ACCOUNT_ID or OANDA_API_KEY/OANDA_ACCOUNT_ID.")
        return

    url = f"{stream_url.rstrip('/')}/v3/accounts/{account_id}/pricing/stream"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept-Datetime-Format": "RFC3339",
    }
    params = {"instruments": instrument, "snapshot": "true"}

    while state.running:
        try:
            state.log(f"Connecting OANDA pricing stream {instrument}")
            with requests.get(url, headers=headers, params=params, stream=True, timeout=30) as r:
                r.raise_for_status()
                state.log(f"Subscribed OANDA {instrument}")

                for line in r.iter_lines():
                    if not state.running:
                        break
                    if not line:
                        continue

                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue

                    if msg.get("type") == "HEARTBEAT":
                        continue
                    if msg.get("type") != "PRICE":
                        continue

                    bids = msg.get("bids") or []
                    asks = msg.get("asks") or []
                    if not bids or not asks:
                        continue

                    bid = safe_float(bids[0].get("price"))
                    ask = safe_float(asks[0].get("price"))
                    mid = float_mid(bid, ask)
                    if mid is None:
                        continue

                    raw_time = msg.get("time")
                    try:
                        tick_time = datetime.fromisoformat(raw_time.replace("Z", "+00:00")).replace(tzinfo=None) if raw_time else datetime.now()
                    except Exception:
                        tick_time = datetime.now()

                    with state.lock:
                        state.latest_oanda_bid = bid
                        state.latest_oanda_ask = ask
                        state.latest_oanda_mid = mid
                        state.latest_oanda_time = tick_time

        except Exception as e:
            state.log(f"OANDA stream error: {e}; reconnecting in 5s")
            time.sleep(5)


async def stream_oanda_spot(state: MarketState, api_key: str, account_id: str, stream_url: str) -> None:
    await asyncio.to_thread(stream_oanda_spot_blocking, state, api_key, account_id, stream_url)


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
    s = (raw_symbol or "").strip().upper().replace("/", ":")
    if not s:
        return BITFINEX_SYMBOL
    if s in {"XAUTUST", "XAUTUSDT"}:
        return "tXAUT:UST"
    if s in {"XAUT:UST", "XAUT:USDT", "XAUT/USD", "XAUT:USD"}:
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
                symbol = normalize_bitfinex_symbol(state.bitfinex_symbol.strip())
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
                            state.log(f"Bitfinex subscription confirmed: {msg.get('symbol') or symbol}")
                            continue
                        if event == "error":
                            state.log(f"Bitfinex error for {symbol}: {msg}")
                            await asyncio.sleep(10)
                            break
                        continue

                    if not isinstance(msg, list) or len(msg) < 2:
                        continue
                    payload = msg[1]
                    if payload == "hb" or not isinstance(payload, list) or len(payload) < 10:
                        continue

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


async def stream_bybit_spot(state: MarketState) -> None:
    """Stream Bybit XAUT/USDT and USDC/USDT top-of-book using orderbook.1."""
    while state.running:
        try:
            async with websockets.connect(
                BYBIT_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1000,
            ) as ws:
                xaut_symbol = state.bybit_xaut_symbol.strip().upper()
                usdc_symbol = state.bybit_usdc_symbol.strip().upper()
                topics = [f"orderbook.1.{xaut_symbol}", f"orderbook.1.{usdc_symbol}"]
                await ws.send(json.dumps({"op": "subscribe", "args": topics}))
                state.log(f"Subscribed Bybit {', '.join(topics)}")

                async for raw_msg in ws:
                    if not state.running:
                        break
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("op") == "subscribe":
                        if not msg.get("success", False):
                            state.log(f"Bybit subscription response: {msg}")
                        continue
                    if msg.get("op") == "ping" or msg.get("op") == "pong":
                        continue

                    topic = msg.get("topic") or ""
                    data = msg.get("data") or {}
                    bids = data.get("b") or []
                    asks = data.get("a") or []
                    if not bids or not asks:
                        continue

                    bid = safe_float(bids[0][0])
                    bid_size = safe_float(bids[0][1])
                    ask = safe_float(asks[0][0])
                    ask_size = safe_float(asks[0][1])
                    mid = float_mid(bid, ask)
                    ts_ms = safe_float(msg.get("ts"))
                    tick_time = datetime.fromtimestamp(ts_ms / 1000) if ts_ms else datetime.now()
                    if mid is None:
                        continue

                    with state.lock:
                        if topic.endswith(f".{xaut_symbol}"):
                            state.quotes["Bybit"] = VenueQuote(bid, ask, mid, None, bid_size, ask_size, tick_time)
                        elif topic.endswith(f".{usdc_symbol}"):
                            state.bybit_usdc_quote = VenueQuote(bid, ask, mid, None, bid_size, ask_size, tick_time)

        except Exception as e:
            state.log(f"Bybit websocket error: {e}; reconnecting in 5s")
            await asyncio.sleep(5)


async def aggregate_rows(state: MarketState, interval_sec: float) -> None:
    while state.running:
        with state.lock:
            gc_mid = state.latest_gc_mid
            gc_bid = state.latest_gc_bid
            gc_ask = state.latest_gc_ask
            oanda_mid = state.latest_oanda_mid
            oanda_bid = state.latest_oanda_bid
            oanda_ask = state.latest_oanda_ask
            bybit_usdc_mid = state.bybit_usdc_quote.mid
            quotes = {k: v for k, v in state.quotes.items()}

            if gc_mid is not None or oanda_mid is not None or any(q.mid is not None for q in quotes.values()):
                row = {
                    "time": datetime.now(),
                    "gc_mid": gc_mid,
                    "gc_bid": gc_bid,
                    "gc_ask": gc_ask,
                    "oanda_spot_mid": oanda_mid,
                    "oanda_spot_bid": oanda_bid,
                    "oanda_spot_ask": oanda_ask,
                    "bybit_usdcusdt_mid": bybit_usdc_mid,
                    "bybit_usdcusdt_bid": state.bybit_usdc_quote.bid,
                    "bybit_usdcusdt_ask": state.bybit_usdc_quote.ask,
                }

                for venue, q in quotes.items():
                    prefix = venue.lower()
                    norm_mid = normalized_xaut_price(q.mid, bybit_usdc_mid)
                    norm_bid = normalized_xaut_price(q.bid, bybit_usdc_mid)
                    norm_ask = normalized_xaut_price(q.ask, bybit_usdc_mid)

                    row[f"{prefix}_mid"] = q.mid
                    row[f"{prefix}_bid"] = q.bid
                    row[f"{prefix}_ask"] = q.ask
                    row[f"{prefix}_last"] = q.last
                    row[f"{prefix}_normalized_mid"] = norm_mid
                    row[f"{prefix}_normalized_bid"] = norm_bid
                    row[f"{prefix}_normalized_ask"] = norm_ask
                    row[f"{prefix}_premium_vs_future_usd"] = norm_mid - gc_mid if norm_mid is not None and gc_mid is not None else None
                    row[f"{prefix}_premium_vs_future_bps"] = premium_bps(norm_mid, gc_mid)
                    row[f"{prefix}_premium_vs_spot_usd"] = norm_mid - oanda_mid if norm_mid is not None and oanda_mid is not None else None
                    row[f"{prefix}_premium_vs_spot_bps"] = premium_bps(norm_mid, oanda_mid)

                state.rows.append(row)

        await asyncio.sleep(interval_sec)


async def run_streams(
    state: MarketState,
    client_secret: str,
    refresh_token: str,
    interval_sec: float,
    oanda_api_key: str,
    oanda_account_id: str,
    oanda_stream_url: str,
) -> None:
    await asyncio.gather(
        stream_comex(state, client_secret, refresh_token),
        stream_oanda_spot(state, oanda_api_key, oanda_account_id, oanda_stream_url),
        stream_bitget_xaut(state),
        stream_bitfinex_xaut(state),
        stream_gate_xaut(state),
        stream_okx_xaut(state),
        stream_bybit_spot(state),
        aggregate_rows(state, interval_sec),
    )


def background_runner(
    state: MarketState,
    client_secret: str,
    refresh_token: str,
    interval_sec: float,
    oanda_api_key: str,
    oanda_account_id: str,
    oanda_stream_url: str,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_streams(
            state,
            client_secret,
            refresh_token,
            interval_sec,
            oanda_api_key,
            oanda_account_id,
            oanda_stream_url,
        ))
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


def start_streaming(
    state: MarketState,
    client_secret: str,
    refresh_token: str,
    interval_sec: float,
    oanda_api_key: str,
    oanda_account_id: str,
    oanda_stream_url: str,
) -> None:
    if state.running:
        return
    if not client_secret or not refresh_token:
        state.log("Missing TT credentials. Set TT_SECRET and TT_REFRESH as environment variables or Streamlit secrets.")
        return

    state.running = True
    state.started_at = datetime.now()

    t = threading.Thread(
        target=background_runner,
        args=(state, client_secret, refresh_token, interval_sec, oanda_api_key, oanda_account_id, oanda_stream_url),
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
    bybit_xaut_symbol: str,
    bybit_usdc_symbol: str,
    max_points: int,
) -> None:
    """Update the shared engine configuration."""
    with state.lock:
        state.gc_symbol = gc_symbol.strip()
        state.bitget_inst_id = bitget_inst_id.strip().upper()
        state.bitfinex_symbol = bitfinex_symbol.strip()
        state.gate_pair = gate_pair.strip().upper()
        state.okx_inst_id = okx_inst_id.strip().upper()
        state.bybit_xaut_symbol = bybit_xaut_symbol.strip().upper()
        state.bybit_usdc_symbol = bybit_usdc_symbol.strip().upper()
        state.rows = deque(state.rows, maxlen=int(max_points))


def restart_streaming(
    state: MarketState,
    client_secret: str,
    refresh_token: str,
    interval_sec: float,
    oanda_api_key: str,
    oanda_account_id: str,
    oanda_stream_url: str,
) -> None:
    if state.running:
        stop_streaming(state)
        time.sleep(1.0)
    start_streaming(state, client_secret, refresh_token, interval_sec, oanda_api_key, oanda_account_id, oanda_stream_url)


# =============================================================================
# UI
# =============================================================================

require_xaut_page_password()

st.title("XAUT Markets vs. Gold Futures & CFD Spot")
st.caption(
    "Live comparison using COMEX, OANDA XAU/USD spot CFD, native exchange websockets for XAUT/USDT, "
    "and Bybit USDC/USDT to normalize USDT-quoted XAUT prices."
)

state = get_state()

client_secret = get_secret_any("TT_SECRET", "TT_CLIENT_SECRET", default="")
refresh_token = get_secret_any("TT_REFRESH", "TT_REFRESH_TOKEN", default="")
oanda_api_key = get_secret_any("OANDA_DEMO_API_KEY", "OANDA_API_KEY", default="")
oanda_account_id = get_secret_any("OANDA_DEMO_ACCOUNT_ID", "OANDA_ACCOUNT_ID", default="")
oanda_stream_url = get_secret_any("OANDA_STREAM_URL", default=OANDA_DEFAULT_STREAM_URL)

with st.sidebar:
    st.header("XAUT vs. Gold Settings")
    st.caption("One shared market-data engine is cached per running Streamlit process. All users read from the same websocket collector.")

    gc_symbol = st.text_input("COMEX symbol", value=state.gc_symbol)
    bitget_inst_id = st.text_input("Bitget instId", value=state.bitget_inst_id, help="Bitget spot symbols are compact, e.g. XAUTUSDT.")
    bitfinex_symbol = st.text_input("Bitfinex symbol", value=state.bitfinex_symbol, help="Default is tXAUT:UST. Bitfinex requires the leading t prefix and a colon for XAUT/USDt.")
    gate_pair = st.text_input("Gate pair", value=state.gate_pair, help="Gate spot pairs use underscore format, e.g. XAUT_USDT.")
    okx_inst_id = st.text_input("OKX instId", value=state.okx_inst_id, help="OKX spot symbols use dash format, e.g. XAUT-USDT.")
    bybit_xaut_symbol = st.text_input("Bybit XAUT symbol", value=state.bybit_xaut_symbol, help="Bybit spot symbols are compact, e.g. XAUTUSDT.")
    bybit_usdc_symbol = st.text_input("Bybit USDC/USDT symbol", value=state.bybit_usdc_symbol, help="Used as the USDT normalization factor. Default: USDCUSDT.")

    max_points = st.number_input("Max stored points", min_value=100, max_value=10000, value=DEFAULT_MAX_POINTS, step=100)
    interval_sec = st.number_input("Aggregation interval seconds", min_value=0.5, max_value=30.0, value=DEFAULT_INTERVAL_SEC, step=0.5)
    refresh_ms = st.number_input("Dashboard refresh ms", min_value=1000, max_value=30000, value=DEFAULT_REFRESH_MS, step=1000)

    apply_restart = st.button("Apply settings & restart shared engine", width="stretch")

    st.divider()
    st.caption(f"Bitget WS: `{BITGET_WS_URL}`")
    st.caption(f"Bitfinex WS: `{BITFINEX_WS_URL}`")
    st.caption(f"Gate WS: `{GATE_WS_URL}`")
    st.caption(f"OKX WS: `{OKX_WS_URL}`")
    st.caption(f"Bybit WS: `{BYBIT_WS_URL}`")
    st.caption(f"OANDA Stream URL: `{oanda_stream_url}`")

if apply_restart:
    configure_state(state, gc_symbol, bitget_inst_id, bitfinex_symbol, gate_pair, okx_inst_id, bybit_xaut_symbol, bybit_usdc_symbol, int(max_points))
    restart_streaming(state, client_secret, refresh_token, float(interval_sec), oanda_api_key, oanda_account_id, oanda_stream_url)
else:
    configure_state(
        state,
        state.gc_symbol,
        state.bitget_inst_id,
        state.bitfinex_symbol,
        state.gate_pair,
        state.okx_inst_id,
        state.bybit_xaut_symbol,
        state.bybit_usdc_symbol,
        int(max_points),
    )
    start_streaming(state, client_secret, refresh_token, float(interval_sec), oanda_api_key, oanda_account_id, oanda_stream_url)


# =============================================================================
# Render dashboard
# =============================================================================

with state.lock:
    rows = list(state.rows)
    logs = list(state.logs)
    latest_quotes = {k: v for k, v in state.quotes.items()}
    latest_usdc = state.bybit_usdc_quote
    latest = {
        "gc_bid": state.latest_gc_bid,
        "gc_ask": state.latest_gc_ask,
        "gc_mid": state.latest_gc_mid,
        "gc_time": state.latest_gc_time,
        "oanda_bid": state.latest_oanda_bid,
        "oanda_ask": state.latest_oanda_ask,
        "oanda_mid": state.latest_oanda_mid,
        "oanda_time": state.latest_oanda_time,
        "running": state.running,
        "started_at": state.started_at,
        "days_to_expiry": days_to_expiry(state.gc_symbol),
        "contract_month": gc_contract_month_name(state.gc_symbol),
        "expiry_date": gc_expiry_date(state.gc_symbol),
    }

df = pd.DataFrame(rows)

status = "RUNNING" if latest["running"] else "STOPPED"
st.subheader(f"Status: {status}")

metric_container = st.container()

with metric_container:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    
with c1:
    contract_month = latest.get("contract_month") or ""
    gc_mid_label = f"COMEX GC {contract_month} Mid" if contract_month else "COMEX GC Contract Mid"
    st.metric(gc_mid_label, f"{latest['gc_mid']:,.2f}" if latest["gc_mid"] is not None else "—")
with c2:
    st.metric("Gold Spot CFD Price", f"{latest['oanda_mid']:,.2f}" if latest["oanda_mid"] is not None else "—")
with c3:
    st.metric("Bybit USDC/USDT", f"{latest_usdc.mid:,.6f}" if latest_usdc.mid is not None else "—")
with c4:
    st.metric("Days to Expiry", f"{latest['days_to_expiry']:,}" if latest.get("days_to_expiry") is not None else "—")


st.markdown("**Exchange XAUT/USDT mids — raw and USDC/USDT-normalized**")
venue_cols = st.columns(len(VENUES))
for col, venue in zip(venue_cols, VENUES):
    q = latest_quotes.get(venue, VenueQuote())
    norm_mid = normalized_xaut_price(q.mid, latest_usdc.mid)
    with col:
        st.metric(venue, f"{norm_mid:,.4f}" if norm_mid is not None else "—")
        if q.mid is not None:
            st.caption(f"raw: {q.mid:,.4f}")
        if norm_mid is not None and latest["oanda_mid"] is not None:
            st.caption(f"vs spot: {premium_bps(norm_mid, latest['oanda_mid']):,.2f} bps")
        elif norm_mid is not None and latest["gc_mid"] is not None:
            st.caption(f"vs GC future: {premium_bps(norm_mid, latest['gc_mid']):,.2f} bps")
        else:
            st.caption("premium: —")

info_cols = st.columns([1, 1, 1])
with info_cols[0]:
    st.metric("Approx. Expiry", latest["expiry_date"].isoformat() if latest.get("expiry_date") else "—")
with info_cols[1]:
    st.metric("OANDA Bid / Ask", f"{latest['oanda_bid']:,.2f} / {latest['oanda_ask']:,.2f}" if latest["oanda_bid"] is not None and latest["oanda_ask"] is not None else "—")
with info_cols[2]:
    st.metric("COMEX Bid / Ask", f"{latest['gc_bid']:,.2f} / {latest['gc_ask']:,.2f}" if latest["gc_bid"] is not None and latest["gc_ask"] is not None else "—")

quote_rows = []
for venue, q in latest_quotes.items():
    norm_mid = normalized_xaut_price(q.mid, latest_usdc.mid)
    quote_rows.append({
        "Venue": venue,
        "Raw Bid": q.bid,
        "Raw Ask": q.ask,
        "Raw Mid": q.mid,
        "Normalized Mid": norm_mid,
        "Bid Size": q.bid_size,
        "Ask Size": q.ask_size,
        "Last Update": q.ts.strftime("%H:%M:%S") if q.ts else None,
        "Premium vs Spot (bps)": premium_bps(norm_mid, latest["oanda_mid"]),
        "Premium vs FM Future (bps)": premium_bps(norm_mid, latest["gc_mid"]),
    })

quote_rows.append({
    "Venue": "Bybit USDC/USDT normalization",
    "Raw Bid": latest_usdc.bid,
    "Raw Ask": latest_usdc.ask,
    "Raw Mid": latest_usdc.mid,
    "Normalized Mid": None,
    "Bid Size": latest_usdc.bid_size,
    "Ask Size": latest_usdc.ask_size,
    "Last Update": latest_usdc.ts.strftime("%H:%M:%S") if latest_usdc.ts else None,
    "Premium vs Spot (bps)": None,
    "Premium vs FM Future (bps)": None,
})

st.dataframe(pd.DataFrame(quote_rows), width="stretch", hide_index=True)

if latest["gc_time"] or latest["oanda_time"] or any(q.ts for q in latest_quotes.values()):
    venue_times = " | ".join(f"{v}: {q.ts.strftime('%H:%M:%S') if q.ts else '—'}" for v, q in latest_quotes.items())
    st.caption(
        f"Last COMEX update: {latest['gc_time'].strftime('%H:%M:%S') if latest['gc_time'] else '—'} | "
        f"Last OANDA update: {latest['oanda_time'].strftime('%H:%M:%S') if latest['oanda_time'] else '—'} | "
        f"Bybit USDC/USDT: {latest_usdc.ts.strftime('%H:%M:%S') if latest_usdc.ts else '—'} | "
        f"{venue_times}"
    )

if df.empty:
    st.info("Waiting for the shared market-data engine to receive COMEX, OANDA, and XAUT venue quotes")
else:
    df = df.sort_values("time")

    fig_prices = go.Figure()
    if "gc_mid" in df.columns and df["gc_mid"].notna().any():
        fig_prices.add_trace(go.Scatter(x=df["time"], y=df["gc_mid"], mode="lines", name="COMEX GC Future"))
    if "oanda_spot_mid" in df.columns and df["oanda_spot_mid"].notna().any():
        fig_prices.add_trace(go.Scatter(x=df["time"], y=df["oanda_spot_mid"], mode="lines", name="Gold Spot CFD Price"))
    for venue in VENUE_KEYS:
        col = f"{venue}_normalized_mid"
        if col in df.columns and df[col].notna().any():
            fig_prices.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=f"{venue.title()} XAUT normalized"))
    fig_prices.update_layout(
        title="Live Absolute Prices",
        xaxis_title="Time",
        yaxis_title="USD / oz",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_prices, width="stretch")

    fig_spot_prem = go.Figure()
    for venue in VENUE_KEYS:
        col = f"{venue}_premium_vs_spot_bps"
        if col in df.columns and df[col].notna().any():
            fig_spot_prem.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=f"{venue.title()} vs Spot CFD"))
    fig_spot_prem.add_hline(y=0, line_dash="dash")
    fig_spot_prem.update_layout(
        title="XAUT Premium / Discount vs. Spot",
        xaxis_title="Time",
        yaxis_title="bps",
        height=390,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_spot_prem, width="stretch")

    fig_future_prem = go.Figure()
    for venue in VENUE_KEYS:
        col = f"{venue}_premium_vs_future_bps"
        if col in df.columns and df[col].notna().any():
            fig_future_prem.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=f"{venue.title()} vs GC Future"))
    fig_future_prem.add_hline(y=0, line_dash="dash")
    fig_future_prem.update_layout(
        title="XAUT Premium / Discount vs. FM Future",
        xaxis_title="Time",
        yaxis_title="bps",
        height=390,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_future_prem, width="stretch")

    fig_norm = go.Figure()
    if "gc_mid" in df.columns and df["gc_mid"].notna().any():
        base = df["gc_mid"].dropna().iloc[0]
        fig_norm.add_trace(go.Scatter(x=df["time"], y=df["gc_mid"] / base * 100, mode="lines", name="COMEX GC normalized"))
    if "oanda_spot_mid" in df.columns and df["oanda_spot_mid"].notna().any():
        base = df["oanda_spot_mid"].dropna().iloc[0]
        fig_norm.add_trace(go.Scatter(x=df["time"], y=df["oanda_spot_mid"] / base * 100, mode="lines", name="Gold Spot CFD normalized"))
    for venue in VENUE_KEYS:
        col = f"{venue}_normalized_mid"
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
    "XAUT venue prices are normalized as XAUT/USDT × (1 / Bybit USDC/USDT) before comparing to OANDA XAU/USD spot CFD or the selected COMEX front-month future. "
    "The old COMEX implied spot/SOFR calculation has been removed."
)

# Auto-refresh. This keeps the app live without adding streamlit-autorefresh.
time.sleep(float(refresh_ms) / 1000.0)
st.rerun()
