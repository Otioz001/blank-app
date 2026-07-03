import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="MT5 Trading Analytics", layout="wide")

st.title("MT5 Trading Analytics Dashboard")

# ---------------------------------------------------------------
# Upload
# ---------------------------------------------------------------
file = st.file_uploader("Upload MT5 CSV export", type="csv")

if not file:
    st.info("Upload your MT5 trade history CSV to get started.")
    st.stop()

df = pd.read_csv(file)

# ---------------------------------------------------------------
# Cleaning / prep
# ---------------------------------------------------------------
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

# Keep only closing rows ("out") — "in" rows always show Profit = 0
# since the trade hasn't been realized yet. Counting them as trades
# would double the row count and dilute win rate / histograms.
trades = df[df["Direction"] == "out"].copy()

trades["Commission"] = pd.to_numeric(trades.get("Commission", 0), errors="coerce").fillna(0)
trades["Swap"] = pd.to_numeric(trades.get("Swap", 0), errors="coerce").fillna(0)
trades["Profit"] = pd.to_numeric(trades["Profit"], errors="coerce").fillna(0)

# Net profit = trading result + costs (commission and swap are usually
# already negative in MT5 exports)
trades["GrossProfit"] = trades["Profit"]
trades["NetProfit"] = trades["Profit"] + trades["Commission"] + trades["Swap"]
trades["Profit"] = trades["NetProfit"]  # use net profit everywhere below

trades["Trade"] = range(len(trades))
trades["Hour"] = trades["Time"].dt.hour
trades["Weekday"] = trades["Time"].dt.day_name()
trades["Cumulative"] = trades["Profit"].cumsum()
trades["CumulativeGross"] = trades["GrossProfit"].cumsum()
trades["TotalCost"] = trades["Commission"] + trades["Swap"]
trades["CumulativeCost"] = trades["TotalCost"].cumsum()
trades["Rolling100"] = trades["Profit"].rolling(100, min_periods=1).sum()

# Drawdown on the Balance series (from the raw df, since balance updates there)
df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")
df["RunningMax"] = df["Balance"].cummax()
df["Drawdown"] = df["Balance"] - df["RunningMax"]

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
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

loss_threshold = st.sidebar.number_input(
    "Only show trades worse than (e.g. -20)", value=0.0, step=1.0
)

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
    filtered = filtered[
        (filtered["Time"].dt.date >= start) & (filtered["Time"].dt.date <= end)
    ]
if loss_threshold < 0:
    filtered = filtered[filtered["Profit"] <= loss_threshold]

# ---------------------------------------------------------------
# Top stats
# ---------------------------------------------------------------
st.subheader("Summary")

net_profit = trades["Profit"].sum()
wins = trades[trades["Profit"] > 0]
losses = trades[trades["Profit"] < 0]
win_rate = len(wins) / len(trades) * 100 if len(trades) else 0
gross_win = wins["Profit"].sum()
gross_loss = abs(losses["Profit"].sum())
profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan
largest_loss = trades["Profit"].min()
max_drawdown = df["Drawdown"].min()

total_commission = trades["Commission"].sum()
total_swap = trades["Swap"].sum()
total_cost = total_commission + total_swap
gross_total = trades["GrossProfit"].sum()
cost_pct_of_gross = (abs(total_cost) / abs(gross_total) * 100) if gross_total != 0 else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net Profit", f"{net_profit:,.2f}")
c2.metric("Win Rate", f"{win_rate:.1f}%")
c3.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "n/a")
c4.metric("Largest Loss", f"{largest_loss:,.2f}")
c5.metric("Max Drawdown", f"{max_drawdown:,.2f}")

c6, c7, c8 = st.columns(3)
c6.metric("Total Commission", f"{total_commission:,.2f}")
c7.metric("Total Swap", f"{total_swap:,.2f}")
c8.metric(
    "Costs as % of Gross Profit",
    f"{cost_pct_of_gross:.1f}%" if not np.isnan(cost_pct_of_gross) else "n/a"
)

# ---------------------------------------------------------------
# Gross vs Net profit comparison
# ---------------------------------------------------------------
st.subheader("Gross vs Net Profit")
gross_net_fig = go.Figure()
gross_net_fig.add_trace(go.Scatter(
    x=trades["Trade"], y=trades["CumulativeGross"],
    mode="lines", name="Gross (before costs)"
))
gross_net_fig.add_trace(go.Scatter(
    x=trades["Trade"], y=trades["Cumulative"],
    mode="lines", name="Net (after costs)"
))
gross_net_fig.update_layout(
    title="Cumulative Gross vs Net Profit — the gap between the lines is what costs are eating",
    xaxis_title="Trade Number", yaxis_title="Cumulative Profit"
)
st.plotly_chart(gross_net_fig, use_container_width=True)

# ---------------------------------------------------------------
# Commission / swap drag over time
# ---------------------------------------------------------------
st.subheader("Commission & Swap Drag Over Time")
cost_fig = go.Figure()
cost_fig.add_trace(go.Scatter(
    x=trades["Trade"], y=trades["CumulativeCost"],
    mode="lines", fill="tozeroy", name="Cumulative Cost"
))
cost_fig.update_layout(
    title="Cumulative Commission + Swap (running total cost of trading)",
    xaxis_title="Trade Number", yaxis_title="Cumulative Cost"
)
st.plotly_chart(cost_fig, use_container_width=True)

# ---------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------
st.subheader("Equity Curve")
fig = px.line(df, x="Time", y="Balance", title="Balance Over Time")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------
st.subheader("Drawdown")
fig = px.area(df, x="Time", y="Drawdown", title="Drawdown Over Time")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Cumulative profit (pure trading performance, ignores deposits)
# ---------------------------------------------------------------
st.subheader("Cumulative Profit")
fig = px.line(trades, x="Trade", y="Cumulative", title="Cumulative Trading Profit")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Rolling 100-trade profit
# ---------------------------------------------------------------
st.subheader("Rolling Profit (100 Trades)")
fig = px.line(
    trades, x="Trade", y="Rolling100",
    title="Rolling 100-Trade Profit — a downward trend means the edge is fading"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Scatter: profit per trade (main "which trades to avoid" chart)
# ---------------------------------------------------------------
st.subheader("Profit per Trade (filtered)")
fig = px.scatter(
    filtered, x="Trade", y="Profit",
    color="Type", size=filtered["Volume"].abs() if "Volume" in filtered else None,
    hover_data=["Time", "Symbol", "Type", "Price"],
    title="Every dot is a trade — points below zero are losses"
)
fig.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Profit distribution
# ---------------------------------------------------------------
st.subheader("Profit Distribution")
fig = px.histogram(filtered, x="Profit", nbins=80, title="Distribution of Trade Profit")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Biggest losers table
# ---------------------------------------------------------------
st.subheader("Biggest Losing Trades")
losers = trades.sort_values("Profit").head(30)
st.dataframe(
    losers[["Trade", "Time", "Symbol", "Type", "Volume", "Price", "Profit", "Hour", "Weekday"]],
    use_container_width=True,
)

# ---------------------------------------------------------------
# Profit by hour
# ---------------------------------------------------------------
st.subheader("Profit by Hour")
hourly = trades.groupby("Hour")["Profit"].sum().reset_index()
fig = px.bar(hourly, x="Hour", y="Profit", title="Total Profit by Hour of Day")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Profit by weekday
# ---------------------------------------------------------------
st.subheader("Profit by Weekday")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = trades.groupby("Weekday")["Profit"].sum().reindex(weekday_order).dropna().reset_index()
fig = px.bar(weekday, x="Weekday", y="Profit", title="Total Profit by Weekday")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Heatmap: weekday vs hour
# ---------------------------------------------------------------
st.subheader("Heatmap: Weekday vs Hour")
heat = trades.pivot_table(
    index="Weekday", columns="Hour", values="Profit", aggfunc="sum", fill_value=0
).reindex(weekday_order).dropna(how="all")
fig = px.imshow(
    heat, color_continuous_scale="RdYlGn", aspect="auto",
    title="Total Profit — Weekday vs Hour (red = avoid)"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Combined View: Price, Commission, Profit")
from plotly.subplots import make_subplots
import plotly.graph_objects as go

combo = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=("Price at Close", "Commission", "Profit (net)")
)

combo.add_trace(
    go.Scatter(x=trades["Trade"], y=trades["Price"], mode="lines", name="Price"),
    row=1, col=1
)

combo.add_trace(
    go.Bar(x=trades["Trade"], y=trades["Commission"], name="Commission"),
    row=2, col=1
)

profit_colors = ["green" if p >= 0 else "red" for p in trades["Profit"]]
combo.add_trace(
    go.Bar(x=trades["Trade"], y=trades["Profit"], marker_color=profit_colors, name="Profit"),
    row=3, col=1
)

combo.update_layout(height=750, showlegend=False)
combo.update_xaxes(title_text="Trade Number", row=3, col=1)
st.plotly_chart(combo, use_container_width=True)

st.caption(
    "Tip: use the sidebar filters to isolate losing trades, a single symbol, "
    "or a date range, then cross-reference against the hour/weekday heatmap "
    "to find conditions worth excluding from the EA."
)