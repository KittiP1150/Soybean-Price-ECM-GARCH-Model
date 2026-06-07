import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Hedge Fund Dashboard | Pairs Trading", layout="wide")

st.title("ECM-GARCH Pairs Trading Dashboard")
st.markdown("**(Soybean vs Soybean Meal) - Statistical Arbitrage Model**")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/backtest_results.csv", index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("Backtest data is not found")
        return None

df = load_data()

if df is not None:
    total_return = (df['Cumulative_Return'].iloc[-1] - 1) * 100
    
    trading_days = len(df)
    annualized_return = ((1 + (total_return/100)) ** (252 / trading_days) - 1) * 100 if trading_days > 0 else 0
    
    daily_vol = df['Strategy_Return'].std()
    ann_vol = daily_vol * np.sqrt(252)
    sharpe = (annualized_return/100) / ann_vol if ann_vol != 0 else 0
    
    running_max = df['Cumulative_Return'].cummax()
    drawdown = (df['Cumulative_Return'] - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    st.markdown("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{total_return:.2f} %")
    col2.metric("Annualized Return", f"{annualized_return:.2f} %")
    col3.metric("Max Drawdown", f"{max_dd:.2f} %")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Trading Signals & Equity Curve")
    long_entries = df[df['Trade_Signal'] == 1]
    short_entries = df[df['Trade_Signal'] == -1]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("Portfolio Value (Equity Curve)", 
                                        "Spread & GARCH Volatility", 
                                        "Dynamic Z-Score & Signals"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Return'], 
                             line=dict(color='green', width=2), name="Equity Curve"), 
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread'], 
                             line=dict(color='white', width=1), name="Spread (Error Term)"), 
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Dynamic_Z'], 
                             line=dict(color='cyan', width=1), name="Z-Score"), 
                  row=3, col=1)

    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Short Signal")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="lime", row=3, col=1, annotation_text="Long Signal")
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
    
    fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries['Dynamic_Z'],
                             mode='markers', marker=dict(color='lime', size=8, symbol='triangle-up'),
                             name="Long Position"), row=3, col=1)
    fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries['Dynamic_Z'],
                             mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'),
                             name="Short Position"), row=3, col=1)

    fig.update_layout(height=800, template="plotly_dark", hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Raw Backtest Data"):
        st.dataframe(df.tail(50))