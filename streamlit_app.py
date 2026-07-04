import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="MT5 Trading Analytics", layout="wide")
st.title("MT5 Trading Analytics Dashboard")

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# =================================================================
# Upload
# =================================================================
file = st.file_uploader("Upload MT5 CSV export", type="csv")

if not file:
    st.info("Upload your MT5 trade history CSV to get started.")
    st.stop()

df = pd.read_csv(file)
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

# =================================================================
# Trade extraction + commission attribution
# -----------------------------------------------------------------
# Some MT5 exports write one row per closed deal (Direction == "out"
# only, as in this file). Others write a paired "in"/"out" row per
# position, where commission sometimes only appears on the "in" row.
# Handle both without assuming which one you have.
# =================================================================
if "Direction" in df.columns and df["Direction"].nunique() > 1:
    trades = df[df["Direction"] == "out"].copy()
    comm_on_in = df[df["Direction"] == "in"]["Commission"].abs().sum() if "Commission" in df.columns else 0
    comm_on_out = df[df["Direction"] == "out"]["Commission"].abs().sum() if "Commission" in df.columns else 0
    if comm_on_in > comm_on_out and "Order" in df.columns:
        comm_map = df[df["Direction"] == "in"].set_index("Order")["Commission"]
        trades["Commission"] = trades["Order"].map(comm_map).fillna(trades.get("Commission", 0))
        st.warning("Commission detected on 'in' rows — remapped onto matching 'out' rows by Order ID.")
else:
    trades = df.copy()

trades["Commission"] = pd.to_numeric(trades.get("Commission", 0), errors="coerce").fillna(0)
trades["Swap"] = pd.to_numeric(trades.get("Swap", 0), errors="coerce").fillna(0)
trades["GrossProfit"] = pd.to_numeric(trades["Profit"], errors="coerce").fillna(0)
trades["TotalCost"] = trades["Commission"] + trades["Swap"]
trades["Profit"] = trades["GrossProfit"] + trades["TotalCost"]  # net profit, used everywhere below

trades["Trade"] = range(len(trades))
trades["Hour"] = trades["Time"].dt.hour
trades["Weekday"] = trades["Time"].dt.day_name()
trades["Month"] = trades["Time"].dt.to_period("M").astype(str)
trades["Date"] = trades["Time"].dt.date
trades["Win"] = trades["Profit"] > 0
trades["Cumulative"] = trades["Profit"].cumsum()
trades["CumulativeGross"] = trades["GrossProfit"].cumsum()
trades["CumulativeCost"] = trades["TotalCost"].cumsum()
trades["Rolling100"] = trades["Profit"].rolling(100, min_periods=1).sum()

# Trade-level running peak / drawdown (on net cumulative profit, not balance,
# so it reflects the strategy itself rather than deposits/withdrawals)
trades["RunningPeak"] = trades["Cumulative"].cummax()
trades["TradeDrawdown"] = trades["Cumulative"] - trades["RunningPeak"]

# Balance-based drawdown (raw df, includes deposits/withdrawals if any)
if "Balance" in df.columns:
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")
    df["RunningMax"] = df["Balance"].cummax()
    df["Drawdown"] = df["Balance"] - df["RunningMax"]


def profit_factor(series):
    wins = series[series > 0].sum()
    losses = -series[series < 0].sum()
    return wins / losses if losses > 0 else np.nan


# =================================================================
# Sidebar filters (used only in the Trade Explorer tab)
# =================================================================
st.sidebar.header("Filters")
only_losses = st.sidebar.checkbox("Losing trades only")
only_wins = st.sidebar.checkbox("Winning trades only")

symbols = sorted(trades["Symbol"].dropna().unique().tolist())
symbol_filter = st.sidebar.multiselect("Symbol", symbols, default=symbols)

directions = sorted(trades["Type"].dropna().unique().tolist())
direction_filter = st.sidebar.multiselect("Buy / Sell", directions, default=directions)

min_date = trades["Time"].min().date()
max_date = trades["Time"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])

loss_threshold = st.sidebar.number_input("Only show trades worse than (e.g. -20)", value=0.0, step=1.0)

filtered = trades.copy()
if only_losses:
    filtered = filtered[filtered["Profit"] < 0]
if only_wins:
    filtered = filtered[filtered["Profit"] > 0]
if symbol_filter:
    filtered = filtered[filtered["Symbol"].isin(symbol_filter)]
if direction_filter:
    filtered = filtered[filtered["Type"].isin(direction_filter)]
if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["Time"].dt.date >= start) & (filtered["Time"].dt.date <= end)]
if loss_threshold < 0:
    filtered = filtered[filtered["Profit"] <= loss_threshold]

# =================================================================
# Tabs — Performance / Diagnostics / Risk / Explorer
# =================================================================
tab_perf, tab_diag, tab_risk, tab_explore = st.tabs(
    ["📊 Performance Overview", "🎯 Strategy Diagnostics", "⚠️ Risk Analysis", "🔍 Trade Explorer"]
)

# -----------------------------------------------------------------
# TAB 1 — Performance Overview
# -----------------------------------------------------------------
with tab_perf:
    st.subheader("Summary")

    net_profit = trades["Profit"].sum()
    gross_profit = trades["GrossProfit"].sum()
    wins = trades[trades["Profit"] > 0]
    losses = trades[trades["Profit"] < 0]
    win_rate = len(wins) / len(trades) * 100 if len(trades) else np.nan
    pf = profit_factor(trades["Profit"])
    expectancy = trades["Profit"].mean()
    median_profit = trades["Profit"].median()
    avg_win = wins["Profit"].mean() if len(wins) else np.nan
    avg_loss = losses["Profit"].mean() if len(losses) else np.nan
    largest_win = trades["Profit"].max()
    largest_loss = trades["Profit"].min()
    max_dd_trade = trades["TradeDrawdown"].min()
    max_dd_balance = df["Drawdown"].min() if "Drawdown" in df.columns else np.nan
    recovery_factor = net_profit / abs(max_dd_trade) if max_dd_trade < 0 else np.nan
    total_commission = trades["Commission"].sum()
    total_swap = trades["Swap"].sum()
    total_cost = total_commission + total_swap
    cost_pct_of_gross = abs(total_cost) / abs(gross_profit) * 100 if gross_profit != 0 else np.nan
    net_gross_ratio = net_profit / gross_profit if gross_profit != 0 else np.nan
    span_days = max((trades["Time"].max() - trades["Time"].min()).days, 1)
    trades_per_day = len(trades) / span_days

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net Profit", f"{net_profit:,.2f}")
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Profit Factor", f"{pf:.2f}" if not np.isnan(pf) else "n/a")
    c4.metric("Expectancy / Trade", f"{expectancy:,.3f}")
    c5.metric("Recovery Factor", f"{recovery_factor:.2f}" if not np.isnan(recovery_factor) else "n/a")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Avg Win", f"{avg_win:,.2f}" if not np.isnan(avg_win) else "n/a")
    c7.metric("Avg Loss", f"{avg_loss:,.2f}" if not np.isnan(avg_loss) else "n/a")
    c8.metric("Largest Win", f"{largest_win:,.2f}")
    c9.metric("Largest Loss", f"{largest_loss:,.2f}")
    c10.metric("Median Profit", f"{median_profit:,.3f}")

    c11, c12, c13, c14, c15 = st.columns(5)
    c11.metric("Max Drawdown (trades)", f"{max_dd_trade:,.2f}")
    c12.metric("Max Drawdown (balance)", f"{max_dd_balance:,.2f}" if not np.isnan(max_dd_balance) else "n/a")
    c13.metric("Trades / Day", f"{trades_per_day:.1f}")
    c14.metric("Commission % of Gross", f"{cost_pct_of_gross:.1f}%" if not np.isnan(cost_pct_of_gross) else "n/a")
    c15.metric("Net / Gross Ratio", f"{net_gross_ratio:.2f}" if not np.isnan(net_gross_ratio) else "n/a")

    st.divider()

    st.subheader("Gross vs Net Profit")
    st.caption(f"Total commission: {total_commission:,.2f} · Total swap: {total_swap:,.2f} · Total cost: {total_cost:,.2f}")
    gross_net_fig = go.Figure()
    gross_net_fig.add_trace(go.Scatter(x=trades["Trade"], y=trades["CumulativeGross"], mode="lines", name="Gross (before costs)"))
    gross_net_fig.add_trace(go.Scatter(x=trades["Trade"], y=trades["Cumulative"], mode="lines", name="Net (after costs)"))
    gross_net_fig.update_layout(title="Cumulative Gross vs Net Profit — the gap is what costs are eating", xaxis_title="Trade Number", yaxis_title="Cumulative Profit")
    st.plotly_chart(gross_net_fig, use_container_width=True)

    if "Balance" in df.columns:
        st.subheader("Equity Curve")
        fig = px.line(df, x="Time", y="Balance", title="Balance Over Time")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdown (Balance)")
        fig = px.area(df, x="Time", y="Drawdown", title="Drawdown Over Time")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Trading Profit")
    fig = px.line(trades, x="Trade", y="Cumulative", title="Cumulative Net Profit")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------
# TAB 2 — Strategy Diagnostics
# -----------------------------------------------------------------
with tab_diag:
    st.subheader("Hour Analytics (the one table that matters most)")
    hour_stats = trades.groupby("Hour").agg(
        Trades=("Profit", "size"),
        WinRate=("Win", "mean"),
        AvgProfit=("Profit", "mean"),
        TotalProfit=("Profit", "sum"),
    )
    hour_stats["ProfitFactor"] = trades.groupby("Hour")["Profit"].apply(profit_factor)
    hour_stats["WinRate"] = (hour_stats["WinRate"] * 100).round(1)
    hour_stats = hour_stats.round(3).reset_index()
    st.dataframe(hour_stats, use_container_width=True)

    fig = px.bar(hour_stats, x="Hour", y="AvgProfit", title="Average Profit by Hour (per-trade, not total)")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(hour_stats, x="Hour", y="ProfitFactor", title="Profit Factor by Hour (below 1.0 = losing hour)")
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(hour_stats, x="Hour", y="Trades", title="Trades by Hour (overtrading check)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(hour_stats, x="Hour", y="WinRate", title="Win Rate by Hour (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Average Profit by Weekday")
    weekday_stats = trades.groupby("Weekday").agg(
        Trades=("Profit", "size"), AvgProfit=("Profit", "mean"), TotalProfit=("Profit", "sum")
    ).reindex(WEEKDAY_ORDER).dropna(how="all").reset_index()
    fig = px.bar(weekday_stats, x="Weekday", y="AvgProfit", title="Average Profit by Weekday")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(weekday_stats.round(3), use_container_width=True)

    st.divider()
    st.subheader("Profit by Price Zone")
    price_min, price_max = trades["Price"].min(), trades["Price"].max()
    n_bins = st.slider("Number of price zones", 5, 30, 15)
    trades["PriceZone"] = pd.cut(trades["Price"], bins=n_bins)
    zone_stats = trades.groupby("PriceZone").agg(
        Trades=("Profit", "size"), AvgProfit=("Profit", "mean"), WinRate=("Win", "mean")
    )
    zone_stats["ProfitFactor"] = trades.groupby("PriceZone")["Profit"].apply(profit_factor)
    zone_stats["WinRate"] = (zone_stats["WinRate"] * 100).round(1)
    zone_stats = zone_stats.round(3).reset_index()
    zone_stats["PriceZone"] = zone_stats["PriceZone"].astype(str)
    fig = px.bar(zone_stats, x="PriceZone", y="AvgProfit", title="Average Profit by Price Zone")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(zone_stats, use_container_width=True)

    st.divider()
    st.subheader("Monthly Net Profit")
    monthly = trades.groupby("Month")["Profit"].sum().reset_index()
    fig = px.bar(monthly, x="Month", y="Profit", title="Net Profit by Month — spot regime changes")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Heatmap: Weekday vs Hour")
    heat = trades.pivot_table(index="Weekday", columns="Hour", values="Profit", aggfunc="sum", fill_value=0).reindex(WEEKDAY_ORDER).dropna(how="all")
    fig = px.imshow(heat, color_continuous_scale="RdYlGn", aspect="auto", title="Total Profit — Weekday vs Hour (red = avoid)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Rolling Metrics (window = 100 trades)")
    window = st.slider("Rolling window (trades)", 20, 300, 100, key="rolling_window")
    trades["RollingWinRate"] = trades["Win"].rolling(window, min_periods=10).mean() * 100
    trades["RollingPF"] = trades["Profit"].rolling(window, min_periods=10).apply(profit_factor, raw=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(trades, x="Trade", y="RollingWinRate", title=f"Rolling Win Rate ({window} trades)")
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(trades, x="Trade", y="RollingPF", title=f"Rolling Profit Factor ({window} trades)")
        fig.add_hline(y=1, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(trades, x="Trade", y="Rolling100", title="Rolling 100-Trade Profit — a downward trend means the edge is fading")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------
# TAB 3 — Risk Analysis
# -----------------------------------------------------------------
with tab_risk:
    st.subheader("Drawdown (Trade-based)")
    fig = px.area(trades, x="Trade", y="TradeDrawdown", title="Drawdown on Net Cumulative Profit")
    st.plotly_chart(fig, use_container_width=True)

    # Longest drawdown duration + recovery time: consecutive trades spent
    # below the running peak before a new peak is set.
    below_peak = trades["TradeDrawdown"] < 0
    dd_groups = (below_peak != below_peak.shift()).cumsum()
    dd_lengths = trades[below_peak].groupby(dd_groups[below_peak]).size()
    longest_dd = int(dd_lengths.max()) if len(dd_lengths) else 0

    c1, c2 = st.columns(2)
    c1.metric("Longest Drawdown (trades underwater)", longest_dd)
    c2.metric("Current Drawdown (trades)", int(dd_lengths.iloc[-1]) if len(dd_lengths) and below_peak.iloc[-1] else 0)

    st.divider()
    st.subheader("Consecutive Win / Loss Streaks")
    sign = np.where(trades["Profit"] >= 0, 1, -1)
    streak_groups = (pd.Series(sign) != pd.Series(sign).shift()).cumsum()
    streak_len = pd.Series(sign).groupby(streak_groups).transform("count")
    streak_sign = pd.Series(sign).groupby(streak_groups).transform("first")

    win_streak_lengths = streak_len[streak_sign == 1].groupby(streak_groups[streak_sign == 1]).first()
    loss_streak_lengths = streak_len[streak_sign == -1].groupby(streak_groups[streak_sign == -1]).first()

    c1, c2 = st.columns(2)
    c1.metric("Worst Losing Streak", int(loss_streak_lengths.max()) if len(loss_streak_lengths) else 0)
    c2.metric("Best Winning Streak", int(win_streak_lengths.max()) if len(win_streak_lengths) else 0)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(loss_streak_lengths, nbins=int(loss_streak_lengths.max()) if len(loss_streak_lengths) else 1,
                            title="Losing Streak Length Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(win_streak_lengths, nbins=int(win_streak_lengths.max()) if len(win_streak_lengths) else 1,
                            title="Winning Streak Length Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Where the Biggest Losses Cluster")
    worst_n = st.slider("Consider N biggest losing trades", 20, 200, 50)
    biggest_losses = trades.sort_values("Profit").head(worst_n)

    c1, c2, c3 = st.columns(3)
    with c1:
        by_hour = biggest_losses.groupby("Hour").size().reset_index(name="Count").sort_values("Count", ascending=False)
        fig = px.bar(by_hour, x="Hour", y="Count", title=f"Biggest {worst_n} Losses by Hour")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        by_wd = biggest_losses.groupby("Weekday").size().reindex(WEEKDAY_ORDER).dropna().reset_index(name="Count")
        fig = px.bar(by_wd, x="Weekday", y="Count", title=f"Biggest {worst_n} Losses by Weekday")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        by_month = biggest_losses.groupby("Month").size().reset_index(name="Count")
        fig = px.bar(by_month, x="Month", y="Count", title=f"Biggest {worst_n} Losses by Month")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        biggest_losses[["Trade", "Time", "Symbol", "Type", "Volume", "Price", "Profit", "Hour", "Weekday"]],
        use_container_width=True,
    )

# -----------------------------------------------------------------
# TAB 4 — Trade Explorer
# -----------------------------------------------------------------
with tab_explore:
    st.caption("Filters apply from the sidebar.")

    st.subheader("Profit per Trade (filtered)")
    fig = px.scatter(
        filtered, x="Trade", y="Profit", color="Type",
        size=filtered["Volume"].abs() if "Volume" in filtered else None,
        hover_data=["Time", "Symbol", "Type", "Price"],
        title="Every dot is a trade — points below zero are losses",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Profit vs Price")
    fig = px.scatter(filtered, x="Price", y="Profit", color="Type", title="Profit vs Entry Price — look for clustering at price extremes")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Profit Distribution")
    fig = px.histogram(filtered, x="Profit", nbins=80, title="Distribution of Trade Profit")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Commission & Swap Drag Over Time")
    cost_fig = go.Figure()
    cost_fig.add_trace(go.Scatter(x=trades["Trade"], y=trades["CumulativeCost"], mode="lines", fill="tozeroy", name="Cumulative Cost"))
    cost_fig.update_layout(title="Cumulative Commission + Swap", xaxis_title="Trade Number", yaxis_title="Cumulative Cost")
    st.plotly_chart(cost_fig, use_container_width=True)

    st.subheader("Combined View: Price, Commission, Profit")
    combo = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=("Price at Close", "Commission", "Profit (net)"))
    combo.add_trace(go.Scatter(x=trades["Trade"], y=trades["Price"], mode="lines", name="Price"), row=1, col=1)
    combo.add_trace(go.Bar(x=trades["Trade"], y=trades["Commission"], name="Commission"), row=2, col=1)
    profit_colors = ["green" if p >= 0 else "red" for p in trades["Profit"]]
    combo.add_trace(go.Bar(x=trades["Trade"], y=trades["Profit"], marker_color=profit_colors, name="Profit"), row=3, col=1)
    combo.update_layout(height=750, showlegend=False)
    combo.update_xaxes(title_text="Trade Number", row=3, col=1)
    st.plotly_chart(combo, use_container_width=True)

    st.subheader("Biggest Losing Trades")
    losers = trades.sort_values("Profit").head(30)
    st.dataframe(
        losers[["Trade", "Time", "Symbol", "Type", "Volume", "Price", "Profit", "Hour", "Weekday"]],
        use_container_width=True,
    )

    st.caption(
        "Tip: use the sidebar filters to isolate losing trades, a single symbol, or a date "
        "range, then cross-reference against the Hour Analytics table in Strategy "
        "Diagnostics to find conditions worth excluding from the EA."
    )