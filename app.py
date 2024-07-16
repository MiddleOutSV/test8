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

# RSI에 따른 색상 계산 함수
def get_color_for_rsi(rsi):
    if rsi == 50:
        return "rgb(255, 255, 255)"  # 흰색
    elif rsi > 50:
        intensity = (rsi - 50) / 50
        r = int(255)
        g = int(255 * (1 - intensity))
        b = int(255 * (1 - intensity))
        return f"rgb({r}, {g}, {b})"
    else:
        intensity = (50 - rsi) / 50
        r = int(255 * (1 - intensity))
        g = int(255 * (1 - intensity))
        b = int(255)
        return f"rgb({r}, {g}, {b})"

# 데이터 수집
@st.cache_data
def load_data():
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
    return pd.DataFrame(data).T

# RSI 시각화
def show_rsi():
    fig_rsi = go.Figure()
    for ticker in tickers:
        rsi = df.loc[ticker, "RSI"]
        color = get_color_for_rsi(rsi)
        fig_rsi.add_trace(go.Scatter(
            x=[ticker], y=[1],
            mode='markers+text',
            marker=dict(size=50, color=color),
            text=str(int(rsi)),
            textfont=dict(color='black'),
            name=ticker
        ))
    fig_rsi.update_layout(title="RSI 비교", yaxis=dict(showticklabels=False, range=[0, 2]))
    st.plotly_chart(fig_rsi)

# EBITDA 비교
def show_ebitda():
    fig_ebitda = go.Figure(data=[go.Bar(x=tickers, y=df["EBITDA"])])
    fig_ebitda.update_layout(title="EBITDA 비교")
    st.plotly_chart(fig_ebitda)

# Free Cash Flow 비교
def show_fcf():
    fig_fcf = go.Figure(data=[go.Bar(x=tickers, y=df["Free Cash Flow"])])
    fig_fcf.update_layout(title="Free Cash Flow 비교")
    st.plotly_chart(fig_fcf)

# 원 크기 계산 함수
def calculate_circle_sizes(values):
    values = np.array(values)
    values = np.where(values > 0, values, np.nan)
    min_size, max_size = 10, 100
    if np.isnan(values).all():
        return [min_size] * len(values)
    sizes = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
    sizes = sizes * (max_size - min_size) + min_size
    return np.where(np.isnan(sizes), min_size, sizes)

# Total Assets 비교
def show_assets():
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
def show_cap():
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
def show_beta():
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

# 앱 제목
st.title("트럼프 수혜주 비교 분석")

# 데이터 로드
df = load_data()

# 버튼 레이아웃 설정
col1, col2, col3 = st.columns(3)
with col1:
    rsi_button = st.button('RSI')
    total_assets_button = st.button('Total Assets')
with col2:
    ebitda_button = st.button('EBITDA')
    total_cap_button = st.button('Total Capitalization')
with col3:
    fcf_button = st.button('Free Cash Flow')
    beta_button = st.button('Beta')

# 시각화를 표시할 컨테이너
chart_container = st.empty()

# 버튼 클릭에 따른 시각화
if rsi_button:
    with chart_container.container():
        show_rsi()
elif ebitda_button:
    with chart_container.container():
        show_ebitda()
elif fcf_button:
    with chart_container.container():
        show_fcf()
elif total_assets_button:
    with chart_container.container():
        show_assets()
elif total_cap_button:
    with chart_container.container():
        show_cap()
elif beta_button:
    with chart_container.container():
        show_beta()
