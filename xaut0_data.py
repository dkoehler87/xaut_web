# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 14:05:03 2025

@author: DKOEH
"""

import requests
import pandas as pd

COIN_TICKERS_URL = "https://api.coingecko.com/api/v3/coins/{id}/tickers"


def fetch_all_tickers(coin_id: str, headers: dict, **extra_params):
    all_tickers = []
    page = 1
    last_data = {}

    while True:
        params = {"page": page}
        params.update(extra_params)

        resp = requests.get(
            COIN_TICKERS_URL.format(id=coin_id),
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        last_data = data

        tickers = data.get("tickers", [])
        if not tickers:
            break

        all_tickers.extend(tickers)

        # tickers are paginated to 100 items per page
        if len(tickers) < 100:
            break

        page += 1

    return {
        "name": last_data.get("name", coin_id),
        "tickers": all_tickers,
    }


def add_market_share(df: pd.DataFrame, volume_col: str = "Volume (USD)") -> pd.DataFrame:
    out = df.copy()
    out[volume_col] = pd.to_numeric(out[volume_col], errors="coerce")
    total = out[volume_col].sum(skipna=True)

    if total and total > 0:
        out["Market Share"] = out[volume_col] / total
    else:
        out["Market Share"] = 0.0

    return out


def build_xaut0_dataframes(coingecko_api_key: str = "", coin_id: str = "tether-gold-tokens"):
    """
    Returns: cex_df, dex_df, usdt_df, btc_df, final_df
    """
    headers = {"x-cg-demo-api-key": f"{coingecko_api_key}"}

    # --- First pull: build DEX mapping (dex_pair_format=contract_address) ---
    result1 = fetch_all_tickers(coin_id, headers, dex_pair_format="contract_address")
    res_df = pd.json_normalize(result1["tickers"])
    res_df["trading_pair"] = res_df["base"] + "/" + res_df["target"]
    res_df = res_df.rename(
        columns={
            "market.name": "venue",
            "market.identifier": "venue_id",
            "converted_volume.usd": "usd_volume",
        }
    )
    dex_dict = {
        venue: "dex"
        for venue in res_df.loc[res_df["trading_pair"].str.len() > 16, "venue"].unique()
    }

    # --- Second pull: include depth ---
    result2 = fetch_all_tickers(coin_id, headers, dex_pair_format="symbol", depth="true")
    ticker_df = pd.json_normalize(result2["tickers"])
    ticker_df["trading_pair"] = ticker_df["base"] + "/" + ticker_df["target"]
    ticker_df = ticker_df.rename(
        columns={
            "market.name": "venue",
            "market.identifier": "venue_id",
            "converted_volume.usd": "usd_volume",
            "bid_ask_spread_percentage": "bid_ask_spr",
            "cost_to_move_up_usd": "ask_depth_200",
            "cost_to_move_down_usd": "bid_depth_200",
        }
    )
    ticker_df["tob_spread_bps"] = (ticker_df["bid_ask_spr"] * 100).round(2)

    # Required columns (same as your script)
    ticker_df = ticker_df[
        [
            "venue",
            "trading_pair",
            "base",
            "target",
            "last",
            "volume",
            "usd_volume",
            "bid_ask_spr",
            "tob_spread_bps",
            "bid_depth_200",
            "ask_depth_200",
            "trust_score",
            "timestamp",
            "is_anomaly",
            "is_stale",
            "venue_id",
        ]
    ].copy()

    # Map venue type
    ticker_df["venue_type"] = ticker_df["venue"].map(dex_dict).fillna("cex")

    # Sort by volume
    ticker_df = ticker_df.sort_values(["usd_volume"], ascending=False)

    # Format timestamp like you do
    ticker_df["timestamp"] = (
        ticker_df["timestamp"]
        .astype("string")
        .str[:-6]
        .str.replace("T", " ", regex=False)
    )

    final_df = ticker_df[
        [
            "timestamp",
            "venue",
            "trading_pair",
            "base",
            "target",
            "last",
            "volume",
            "usd_volume",
            "tob_spread_bps",
            "bid_depth_200",
            "ask_depth_200",
            "trust_score",
            "venue_type",
        ]
    ].copy()

    final_df.columns = ['Timestamp', 'Venue', 'Trading Pair', 'Base', 'Quote', 'Last', 'Volume', 'Volume (USD)', 'TOB Spread (bps)', 'Bid Depth (200 bps)', 'Ask Depth (200 bps)', 'Trust Score', 'Venue Type']

    # # Split into your 4 outputs
    # cex_df = final_df[final_df["Venue Type"] == "cex"].copy()
    # dex_df = final_df[final_df["Venue Type"] == "dex"].copy()
    # usdt_df = cex_df[cex_df["Quote"] == "USDT"].copy()
    # btc_df = cex_df[(cex_df["Base"] == "BTC") | (cex_df["Quote"] == "BTC")].copy()
    # usd_df = cex_df[cex_df['Quote']=='USD'].copy()

    
    # cex_df = add_market_share(cex_df)
    # dex_df = add_market_share(dex_df)
    # usdt_df = add_market_share(usdt_df)
    # btc_df = add_market_share(btc_df)
    # usd_df = add_market_share(usd_df)

    
    all_df = add_market_share(final_df)


    return all_df

    
    

