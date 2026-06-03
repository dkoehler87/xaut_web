# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:46:57 2026

@author: DKOEH
"""

from io import BytesIO
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Historical Data", layout="wide")

DEFAULT_TOKENS = ["XAUT", "PAXG", "XAUT0", "USAT"]
TOKEN_ORDER = ["XAUT", "PAXG", "XAUT0", "USAT"]

TABLE_NAME = "token_metrics_daily"
SUPABASE_PAGE_SIZE = 1000

TOKEN_BG_COLORS = {
    "XAUT": "rgba(245, 158, 11, 0.18)",
    "XAUT0": "rgba(59, 130, 246, 0.18)",
    "PAXG": "rgba(34, 197, 94, 0.18)",
    "USAT": "rgba(168, 85, 247, 0.18)",
}

TOKEN_BORDER_COLORS = {
    "XAUT": "rgba(245, 158, 11, 0.45)",
    "XAUT0": "rgba(59, 130, 246, 0.45)",
    "PAXG": "rgba(34, 197, 94, 0.45)",
    "USAT": "rgba(168, 85, 247, 0.45)",
}


# -----------------------------
# Secrets loader
# -----------------------------
def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        value = os.getenv(key)
        if value is None:
            raise RuntimeError(f"Missing secret: {key}")
        return value


# -----------------------------
# Formatting helpers
# -----------------------------
def integer_format(x):
    if pd.isna(x):
        return ""
    return f"{round(float(x)):,.0f}"


def price_format(x):
    if pd.isna(x):
        return ""
    return f"{float(x):,.2f}"


def pct_change_format(x):
    if pd.isna(x):
        return "—"
    return f"{x:+.1f}%"


def pct_format_2(x):
    if pd.isna(x):
        return "—"
    return f"{x:.2f}%"


def change_color(x):
    if pd.isna(x):
        return "#9ca3af"
    if x > 0:
        return "#22c55e"
    if x < 0:
        return "#ef4444"
    return "#9ca3af"


def tile_bg(idx: int) -> str:
    colors = [
        "rgba(255,255,255,0.045)",
        "rgba(255,255,255,0.075)",
        "rgba(255,255,255,0.055)",
        "rgba(255,255,255,0.090)",
        "rgba(255,255,255,0.060)",
    ]
    return colors[idx % len(colors)]


def safe_avg_abs_pct(series: pd.Series):
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    return cleaned.abs().mean()


# -----------------------------
# Excel export
# -----------------------------
def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Historical Data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


# -----------------------------
# Supabase loader (pagination)
# -----------------------------
@st.cache_data(ttl=3600)
def load_token_history():
    supabase = create_client(
        get_secret("SUPABASE_URL"),
        get_secret("SUPABASE_KEY"),
    )

    all_rows = []
    start = 0

    while True:
        end = start + SUPABASE_PAGE_SIZE - 1

        response = (
            supabase
            .table(TABLE_NAME)
            .select("date, token, price, volume_usd, market_cap")
            .order("date", desc=False)
            .range(start, end)
            .execute()
        )

        batch = response.data or []

        if not batch:
            break

        all_rows.extend(batch)

        if len(batch) < SUPABASE_PAGE_SIZE:
            break

        start += SUPABASE_PAGE_SIZE

    df = pd.DataFrame(all_rows)

    df["date"] = pd.to_datetime(df["date"])
    df["token"] = df["token"].str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume_usd"] = pd.to_numeric(df["volume_usd"], errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

    return df


# -----------------------------
# Pivot helper
# -----------------------------
def build_metric_pivot(df, metric):
    pivot = (
        df.pivot(index="date", columns="token", values=metric)
        .sort_index(ascending=False)
        .reset_index()
    )

    pivot["date"] = pd.to_datetime(pivot["date"]).dt.date

    ordered_cols = ["date"] + [t for t in TOKEN_ORDER if t in pivot.columns]
    pivot = pivot[ordered_cols]
    pivot = pivot.rename(columns={"date": "Date"})

    return pivot


# -----------------------------
# Snapshot card calculations
# -----------------------------
def build_snapshot_cards(df):
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()

    prev_df = df[df["date"] < latest_date]

    if not prev_df.empty:
        prev = (
            prev_df.sort_values(["token", "date"])
            .groupby("token")
            .tail(1)
        )

        prev = prev.rename(columns={
            "price": "prev_price",
            "volume_usd": "prev_volume",
            "market_cap": "prev_market_cap",
        })

        latest = latest.merge(
            prev[["token", "prev_price", "prev_volume", "prev_market_cap"]],
            on="token",
            how="left"
        )

        latest["price_change_pct"] = (
            (latest["price"] - latest["prev_price"]) / latest["prev_price"] * 100
        )

        latest["volume_change_pct"] = (
            (latest["volume_usd"] - latest["prev_volume"]) / latest["prev_volume"] * 100
        )

        latest["market_cap_change_pct"] = (
            (latest["market_cap"] - latest["prev_market_cap"]) / latest["prev_market_cap"] * 100
        )
    else:
        latest["price_change_pct"] = pd.NA
        latest["volume_change_pct"] = pd.NA
        latest["market_cap_change_pct"] = pd.NA

    latest["token"] = pd.Categorical(
        latest["token"],
        categories=TOKEN_ORDER,
        ordered=True
    )

    latest = latest.sort_values("token")
    return latest


# -----------------------------
# Daily analytics base
# -----------------------------
def build_analytics_base(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df.copy().sort_values(["token", "date"])
    x["prev_price"] = x.groupby("token")["price"].shift(1)
    x["prev_volume"] = x.groupby("token")["volume_usd"].shift(1)
    x["prev_market_cap"] = x.groupby("token")["market_cap"].shift(1)

    x["price_change_abs"] = x["price"] - x["prev_price"]
    x["price_change_pct"] = np.where(
        x["prev_price"].notna() & (x["prev_price"] != 0),
        (x["price"] - x["prev_price"]) / x["prev_price"] * 100,
        np.nan
    )

    x["volume_change_abs"] = x["volume_usd"] - x["prev_volume"]
    x["volume_change_pct"] = np.where(
        x["prev_volume"].notna() & (x["prev_volume"] != 0),
        (x["volume_usd"] - x["prev_volume"]) / x["prev_volume"] * 100,
        np.nan
    )

    x["market_cap_change_pct"] = np.where(
        x["prev_market_cap"].notna() & (x["prev_market_cap"] != 0),
        (x["market_cap"] - x["prev_market_cap"]) / x["prev_market_cap"] * 100,
        np.nan
    )

    x["token"] = pd.Categorical(x["token"], categories=TOKEN_ORDER, ordered=True)
    x = x.sort_values(["date", "token"], ascending=[False, True])

    return x


def build_token_summary(base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame()

    grouped = (
        base_df.groupby("token", observed=True)
        .agg(
            Days=("date", "count"),
            Avg_Price=("price", "mean"),
            Avg_Daily_Move_Pct=("price_change_pct", "mean"),
            Avg_Abs_Move_Pct=("price_change_pct", safe_avg_abs_pct),
            Total_Volume_USD=("volume_usd", "sum"),
            Avg_Volume_USD=("volume_usd", "mean"),
            Avg_Market_Cap=("market_cap", "mean"),
            Up_Days=("price_change_pct", lambda s: (s > 0).sum()),
            Down_Days=("price_change_pct", lambda s: (s < 0).sum()),
        )
        .reset_index()
    )

    grouped = grouped.rename(columns={
        "token": "Token",
        "Avg_Price": "Avg Price",
        "Avg_Daily_Move_Pct": "Avg Daily Move %",
        "Avg_Abs_Move_Pct": "Avg Abs Move %",
        "Total_Volume_USD": "Total Volume USD",
        "Avg_Volume_USD": "Avg Volume USD",
        "Avg_Market_Cap": "Avg Market Cap",
        "Up_Days": "Up Days",
        "Down_Days": "Down Days",
    })

    grouped["Token"] = pd.Categorical(grouped["Token"], categories=TOKEN_ORDER, ordered=True)
    grouped = grouped.sort_values("Token")
    return grouped


def build_grouped_analytics_table(base_df: pd.DataFrame, table_tokens: list[str]) -> pd.DataFrame:
    if base_df.empty or not table_tokens:
        return pd.DataFrame()

    metrics = {
        "price": "Price",
        "price_change_pct": "Daily Price Change %",
        "volume_usd": "Volume USD",
        "volume_change_pct": "Daily Volume Change %",
        "market_cap": "Market Cap",
    }

    wide_pieces = []
    working = base_df[base_df["token"].isin(table_tokens)].copy()

    for token in table_tokens:
        token_df = (
            working[working["token"] == token]
            .sort_values("date", ascending=False)
            .set_index("date")
        )

        token_df = token_df[list(metrics.keys())].rename(columns=metrics)
        token_df.columns = pd.MultiIndex.from_product([[token], token_df.columns])
        wide_pieces.append(token_df)

    if not wide_pieces:
        return pd.DataFrame()

    wide = pd.concat(wide_pieces, axis=1)
    wide.index = pd.to_datetime(wide.index).date
    wide.index.name = "Date"
    wide = wide.sort_index(ascending=False)

    return wide


# -----------------------------
# Styling helpers
# -----------------------------
def style_summary_table(df: pd.DataFrame):
    return df.style.format({
        "Avg Price": price_format,
        "Avg Daily Move %": pct_format_2,
        "Avg Abs Move %": pct_format_2,
        "Total Volume USD": integer_format,
        "Avg Volume USD": integer_format,
        "Avg Market Cap": integer_format,
    })


def style_grouped_analytics_table(df: pd.DataFrame):
    if df.empty:
        return df.style

    pct_cols = []
    price_cols = []
    integer_cols = []

    for col in df.columns:
        metric = col[1]
        if "%" in metric:
            pct_cols.append(col)
        elif metric == "Price":
            price_cols.append(col)
        else:
            integer_cols.append(col)

    styler = df.style.format(
        {col: price_format for col in price_cols} |
        {col: integer_format for col in integer_cols} |
        {col: pct_format_2 for col in pct_cols}
    )

    def color_pct(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #22c55e; font-weight: 600;"
        if val < 0:
            return "color: #ef4444; font-weight: 600;"
        return "color: #9ca3af; font-weight: 600;"

    pct_change_like = [
        col for col in df.columns
        if col[1] in ["Daily Price Change %", "Daily Volume Change %"]
    ]

    if pct_change_like:
        styler = styler.map(color_pct, subset=pct_change_like)

    # Color entire token groups
    for token in df.columns.get_level_values(0).unique():
        token_cols = [col for col in df.columns if col[0] == token]
        bg = TOKEN_BG_COLORS.get(token, "rgba(255,255,255,0.06)")
        border = TOKEN_BORDER_COLORS.get(token, "rgba(255,255,255,0.15)")

        styler = styler.set_properties(
            subset=pd.IndexSlice[:, token_cols],
            **{
                "background-color": bg,
                "border-left": f"1px solid {border}",
                "border-right": f"1px solid {border}",
            },
        )

    # Header styling by exact rendered column positions
    table_styles = []
    cols = list(df.columns)

    for idx, col in enumerate(cols):
        token = col[0]
        bg = TOKEN_BG_COLORS.get(token, "rgba(255,255,255,0.06)")
        border = TOKEN_BORDER_COLORS.get(token, "rgba(255,255,255,0.15)")

        table_styles.append({
            "selector": f"th.col_heading.level0.col{idx}",
            "props": [
                ("background-color", bg),
                ("border-bottom", f"1px solid {border}"),
                ("color", "white"),
            ],
        })
        table_styles.append({
            "selector": f"th.col_heading.level1.col{idx}",
            "props": [
                ("background-color", bg),
                ("border-bottom", f"1px solid {border}"),
                ("color", "white"),
            ],
        })

    styler = styler.set_table_styles(table_styles, overwrite=False)
    return styler


metric_labels = {
    "volume_usd": "Volume USD",
    "price": "Price",
    "market_cap": "Market Cap",
}


# -----------------------------
# Local CSS only
# -----------------------------
st.markdown(
    """
    <style>
    .snapshot-token {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .snapshot-date {
        color: #9ca3af;
        font-size: 0.80rem;
        margin-bottom: 10px;
    }

    .mini-tile {
        border-radius: 12px;
        padding: 10px 12px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 8px;
        background: rgba(255,255,255,0.04);
    }

    .mini-label {
        font-size: 0.76rem;
        color: #9ca3af;
        margin-bottom: 4px;
        line-height: 1.15;
    }

    .mini-value {
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.15;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Page UI
# =====================================================

st.title("Historical Data")

df = load_token_history()

available_tokens = sorted(df["token"].unique())
min_date = df["date"].min().date()
max_date = df["date"].max().date()

# -----------------------------------------------------
# Latest Snapshot
# -----------------------------------------------------
st.subheader("Latest Snapshot")

snapshot = build_snapshot_cards(df)
cols = st.columns(len(snapshot))

for col, row in zip(cols, snapshot.itertuples()):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="snapshot-token">{row.token}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="snapshot-date">Data Date: {row.date.date()}</div>', unsafe_allow_html=True)

            t1, t2 = st.columns(2)
            with t1:
                st.markdown(
                    f"""
                    <div class="mini-tile" style="background:{tile_bg(0)};">
                        <div class="mini-label">Price</div>
                        <div class="mini-value">{price_format(row.price)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with t2:
                st.markdown(
                    f"""
                    <div class="mini-tile" style="background:{tile_bg(1)};">
                        <div class="mini-label">Price Change vs Prior</div>
                        <div class="mini-value" style="color:{change_color(row.price_change_pct)};">
                            {pct_change_format(row.price_change_pct)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            t3, t4 = st.columns(2)
            with t3:
                st.markdown(
                    f"""
                    <div class="mini-tile" style="background:{tile_bg(2)};">
                        <div class="mini-label">Volume USD</div>
                        <div class="mini-value">{integer_format(row.volume_usd)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with t4:
                st.markdown(
                    f"""
                    <div class="mini-tile" style="background:{tile_bg(3)};">
                        <div class="mini-label">Volume Change vs Prior</div>
                        <div class="mini-value" style="color:{change_color(row.volume_change_pct)};">
                            {pct_change_format(row.volume_change_pct)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="mini-tile" style="background:{tile_bg(4)};">
                    <div class="mini-label">Market Cap</div>
                    <div class="mini-value">{integer_format(row.market_cap)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------
# Filters + Summary
# -----------------------------------------------------
st.subheader("Filters")

filters_col, summary_col = st.columns([1.05, 2.6])

with filters_col:
    with st.container(border=True):
        selected_tokens = st.multiselect(
            "Tokens",
            options=available_tokens,
            default=[t for t in DEFAULT_TOKENS if t in available_tokens],
        )

        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        table_mode = st.selectbox(
            "Table View",
            ["analytics", "pivot"],
            format_func=lambda x: "Analytical Daily Table" if x == "analytics" else "Metric Pivot Table",
        )

        table_metric = st.selectbox(
            "Pivot Metric",
            ["volume_usd", "price", "market_cap"],
            format_func=lambda x: metric_labels[x],
            disabled=(table_mode != "pivot"),
        )

        chart_metric = st.selectbox(
            "Chart Metric",
            ["volume_usd", "market_cap", "price"],
            format_func=lambda x: metric_labels[x],
        )

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
else:
    start_date = pd.to_datetime(min_date)
    end_date = pd.to_datetime(max_date)

filtered = df[
    (df["token"].isin(selected_tokens)) &
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
].copy()

if filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

analytics_base_df = build_analytics_base(filtered)
summary_df = build_token_summary(analytics_base_df)

with summary_col:
    st.subheader("Selected Range Summary")
    st.dataframe(
        style_summary_table(summary_df),
        width="stretch",
        hide_index=True,
    )


# -----------------------------------------------------
# Table
# -----------------------------------------------------
if table_mode == "analytics":
    st.subheader("Analytical Daily Table")

    ordered_selected_tokens = [t for t in TOKEN_ORDER if t in selected_tokens]
    default_table_tokens = ordered_selected_tokens[: min(2, len(ordered_selected_tokens))]

    table_tokens = st.multiselect(
        "Analytical Table Tokens (up to 2)",
        options=ordered_selected_tokens,
        default=default_table_tokens,
        max_selections=2,
    )

    grouped_table_df = build_grouped_analytics_table(analytics_base_df, table_tokens)

    if grouped_table_df.empty:
        st.info("Select up to 2 tokens to display the grouped analytical table.")
    else:
        st.dataframe(
            style_grouped_analytics_table(grouped_table_df),
            width="stretch",
        )

        analytics_excel = dataframe_to_excel_bytes(
            grouped_table_df,
            sheet_name="Grouped Analytics Table"
        )

        st.download_button(
            "Download Analytical Table as Excel",
            data=analytics_excel,
            file_name="historical_grouped_analytics_table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.subheader(f"{metric_labels[table_metric]} Pivot Table")

    pivot_df = build_metric_pivot(filtered, table_metric)

    format_dict = {
        col: integer_format if table_metric != "price" else price_format
        for col in pivot_df.columns if col != "Date"
    }

    st.dataframe(
        pivot_df.style.format(format_dict),
        width="stretch",
        hide_index=True,
    )

    excel_bytes = dataframe_to_excel_bytes(pivot_df, sheet_name="Pivot Table")

    st.download_button(
        "Download Pivot Table as Excel",
        data=excel_bytes,
        file_name=f"historical_{table_metric}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------------------------------------
# Chart
# -----------------------------------------------------
st.subheader(f"{metric_labels[chart_metric]} by Token")

if chart_metric == "price":
    chart = (
        alt.Chart(filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("price:Q", title="Price"),
            color=alt.Color("token:N", title="Token"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("token:N", title="Token"),
                alt.Tooltip("price:Q", title="Price", format=",.2f"),
                alt.Tooltip("volume_usd:Q", title="Volume USD", format=",.0f"),
                alt.Tooltip("market_cap:Q", title="Market Cap", format=",.0f"),
            ],
        )
        .properties(height=450)
        .interactive()
    )
else:
    chart = (
        alt.Chart(filtered)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{chart_metric}:Q", title=metric_labels[chart_metric]),
            color=alt.Color("token:N", title="Token"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("token:N", title="Token"),
                alt.Tooltip(f"{chart_metric}:Q", title=metric_labels[chart_metric], format=",.0f"),
                alt.Tooltip("price:Q", title="Price", format=",.2f"),
                alt.Tooltip("volume_usd:Q", title="Volume USD", format=",.0f"),
            ],
        )
        .properties(height=450)
        .interactive()
    )

st.altair_chart(chart, width="stretch")
