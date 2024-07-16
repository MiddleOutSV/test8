import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 트럼프 수혜주 티커 리스트
tickers = ["DJT", "GEO", "AXON", "CDMO", "BRCC", "INTC"]

# 데이터 가져오기
@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period="1y")
    return info, history

# RSI 계산 함수
def calculate_rsi(data, periods=14):
    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi.iloc[-1]

# 앱 제목
st.title("트럼프 수혜주 비교 분석")

# 데이터 수집
data = {}
for ticker in tickers:
    info, history = get_stock_data(ticker)
    rsi = calculate_rsi(history)
    data[ticker] = {
        "RSI": rsi,
        "EBITDA": info.get("ebitda", 0) or 0,
        "Total Assets": info.get("totalAssets", 0) or 0,
        "Total Capitalization": info.get("marketCap", 0) or 0,
        "Free Cash Flow": info.get("freeCashflow", 0) or 0,
        "Beta": info.get("beta", 0) or 0
    }

df = pd.DataFrame(data).T

# RSI 시각화
fig_rsi = go.Figure()
for ticker in tickers:
    rsi = df.loc[ticker, "RSI"]
    color = f"rgb({int(255 * (rsi - 50) / 50) if rsi > 50 else 0}, 0, {int(255 * (50 - rsi) / 50) if rsi < 50 else 0})"
    fig_rsi.add_trace(go.Scatter(
        x=[ticker], y=[1],
        mode='markers+text',
        marker=dict(size=50, color=color),
        text=str(int(rsi)),
        textfont=dict(color='white'),
        name=ticker
    ))

fig_rsi.update_layout(title="RSI 비교", yaxis=dict(showticklabels=False, range=[0, 2]))
st.plotly_chart(fig_rsi)

# EBITDA 비교
fig_ebitda = go.Figure(data=[go.Bar(x=tickers, y=df["EBITDA"])])
fig_ebitda.update_layout(title="EBITDA 비교")
st.plotly_chart(fig_ebitda)

# Free Cash Flow 비교
fig_fcf = go.Figure(data=[go.Bar(x=tickers, y=df["Free Cash Flow"])])
fig_fcf.update_layout(title="Free Cash Flow 비교")
st.plotly_chart(fig_fcf)

# 원 크기 계산 함수
def calculate_circle_sizes(values):
    values = np.array(values)
    values = np.where(values > 0, values, np.nan)  # 0 이하의 값을 NaN으로 변경
    min_size, max_size = 10, 100  # 최소 및 최대 원 크기
    if np.isnan(values).all():
        return [min_size] * len(values)
    sizes = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
    sizes = sizes * (max_size - min_size) + min_size
    return np.where(np.isnan(sizes), min_size, sizes)

# Total Assets 비교
asset_sizes = calculate_circle_sizes(df["Total Assets"])
fig_assets = go.Figure(data=[go.Scatter(
    x=tickers,
    y=[1]*len(tickers),
    mode='markers+text',
    marker=dict(size=asset_sizes, sizemode='diameter'),
    text=df["Total Assets"].apply(lambda x: f"{x:,.0f}"),
    textposition='top center',
    hoverinfo='text'
)])
fig_assets.update_layout(title="Total Assets 비교", yaxis=dict(showticklabels=False, range=[0, 2]))
st.plotly_chart(fig_assets)

# Total Capitalization 비교
cap_sizes = calculate_circle_sizes(df["Total Capitalization"])
fig_cap = go.Figure(data=[go.Scatter(
    x=tickers,
    y=[1]*len(tickers),
    mode='markers+text',
    marker=dict(size=cap_sizes, sizemode='diameter'),
    text=df["Total Capitalization"].apply(lambda x: f"{x:,.0f}"),
    textposition='top center',
    hoverinfo='text'
)])
fig_cap.update_layout(title="Total Capitalization 비교", yaxis=dict(showticklabels=False, range=[0, 2]))
st.plotly_chart(fig_cap)

# Beta 비교
beta_sizes = calculate_circle_sizes(df["Beta"])
fig_beta = go.Figure(data=[go.Scatter(
    x=tickers,
    y=[1]*len(tickers),
    mode='markers+text',
    marker=dict(size=beta_sizes, sizemode='diameter'),
    text=df["Beta"].apply(lambda x: f"{x:.2f}"),
    textposition='top center',
    hoverinfo='text'
)])
fig_beta.update_layout(title="Beta 비교", yaxis=dict(showticklabels=False, range=[0, 2]))
st.plotly_chart(fig_beta)
