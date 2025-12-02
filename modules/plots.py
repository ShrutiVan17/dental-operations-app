# modules/plots.py
"""
Plot helpers for the Dental Scheduler dashboard.

All functions return Plotly figures that are friendly for a non-technical owner:
- Clear colours
- Simple titles
- Useful hovers
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# A small colour palette that works on dark background
COLOR_MAIN = "#38bdf8"   # cyan
COLOR_ACCENT = "#f97316" # orange
COLOR_MID = "#6366f1"    # indigo
COLOR_SOFT = "#22c55e"   # green


def _clean_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    return out.sort_values(date_col)


# ---------------------------------------------------------------------
# 1. Time-series line: metric over time + 7-day moving average
# ---------------------------------------------------------------------
def timeseries_line(df: pd.DataFrame, date_col: str, value_col: str, title: str):
    if df.empty or value_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available", template="plotly_dark")
        return fig

    df_ts = _clean_date(df[[date_col, value_col]], date_col)
    if df_ts.empty:
        fig = go.Figure()
        fig.update_layout(title="No valid dates in data", template="plotly_dark")
        return fig

    # daily aggregation so we don't double-count
    daily = (
        df_ts.groupby(date_col, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "value"})
    )
    # 7-day rolling average (only if enough points)
    if len(daily) >= 3:
        daily["rolling_7"] = daily["value"].rolling(window=7, min_periods=1).mean()
    else:
        daily["rolling_7"] = daily["value"]

    fig = go.Figure()

    # Raw daily values with markers
    fig.add_trace(
        go.Scatter(
            x=daily[date_col],
            y=daily["value"],
            mode="lines+markers",
            name="Daily collections",
            line=dict(color=COLOR_MAIN, width=2),
            marker=dict(size=4),
            hovertemplate="%{x|%b %d, %Y}<br>Value: $%{y:,.0f}<extra></extra>",
        )
    )

    # Rolling average line
    fig.add_trace(
        go.Scatter(
            x=daily[date_col],
            y=daily["rolling_7"],
            mode="lines",
            name="7-day avg",
            line=dict(color=COLOR_ACCENT, width=3, dash="solid"),
            hovertemplate="7-day avg: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title=value_col,
        xaxis_title="Date",
    )
    return fig


# ---------------------------------------------------------------------
# 2. Year/Month heatmap: where are we strongest?
# ---------------------------------------------------------------------
def year_month_breakdown(df: pd.DataFrame, date_col: str, value_col: str, title: str):
    if df.empty or value_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available", template="plotly_dark")
        return fig

    df_tmp = _clean_date(df[[date_col, value_col]], date_col)
    if df_tmp.empty:
        fig = go.Figure()
        fig.update_layout(title="No valid dates in data", template="plotly_dark")
        return fig

    df_tmp["year"] = df_tmp[date_col].dt.year
    df_tmp["month_num"] = df_tmp[date_col].dt.month
    df_tmp["month"] = df_tmp[date_col].dt.month_name().str.slice(stop=3)

    pivot = (
        df_tmp.groupby(["year", "month_num", "month"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "value"})
    )

    # make a matrix: rows = year, columns = month (1–12)
    table = pivot.pivot_table(
        index="year", columns="month_num", values="value", fill_value=0
    ).sort_index()

    # pretty month labels
    month_labels = [
        pivot.loc[pivot["month_num"] == m, "month"].iloc[0]
        if (pivot["month_num"] == m).any()
        else ""
        for m in table.columns
    ]

    fig = px.imshow(
        table,
        labels=dict(x="Month", y="Year", color=value_col),
        x=month_labels,
        y=table.index,
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=0, r=60, t=40, b=0),
    )
    fig.update_traces(
        hovertemplate="Year %{y}, %{x}<br>" + value_col + ": $%{z:,.0f}<extra></extra>"
    )
    return fig


# ---------------------------------------------------------------------
# 3. Generic bar chart for breakdowns
# ---------------------------------------------------------------------
def bar_by(df: pd.DataFrame, group_col: str, value_col: str, title: str):
    if df.empty or value_col not in df.columns or group_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available", template="plotly_dark")
        return fig

    df_tmp = df[[group_col, value_col]].copy()
    df_tmp[value_col] = pd.to_numeric(df_tmp[value_col], errors="coerce")
    df_tmp = df_tmp.dropna(subset=[value_col])

    # Special handling for weekday to keep 0–6 order & friendly labels
    if group_col == "dayofweek":
        weekday_map = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
        df_tmp[group_col] = df_tmp[group_col].map(weekday_map)
        category_order = list(weekday_map.values())
    elif group_col == "month":
        month_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        df_tmp[group_col] = df_tmp[group_col].map(month_map)
        category_order = list(month_map.values())
    else:
        category_order = None

    agg = (
        df_tmp.groupby(group_col, as_index=True)[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "value"})
    )

    # Decide orientation: if many categories, go horizontal for readability
    horizontal = group_col in ("year",) or len(agg) > 10

    if horizontal:
        fig = px.bar(
            agg.sort_values("value"),
            x="value",
            y=group_col,
            orientation="h",
            title=title,
            color="value",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            xaxis_title=value_col,
            yaxis_title="",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_traces(
            hovertemplate="%{y}<br>" + value_col + ": $%{x:,.0f}<extra></extra>"
        )
    else:
        fig = px.bar(
            agg,
            x=group_col,
            y="value",
            title=title,
            color="value",
            color_continuous_scale="Blues",
        )
        if category_order is not None:
            fig.update_xaxes(categoryorder="array", categoryarray=category_order)

        fig.update_layout(
            xaxis_title="",
            yaxis_title=value_col,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_traces(
            hovertemplate="%{x}<br>" + value_col + ": $%{y:,.0f}<extra></extra>"
        )

    return fig
