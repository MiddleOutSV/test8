import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
        "EBITDA": info.get("ebitda", 0),
        "Total Assets": info.get("totalAssets", 0),
        "Total Capitalization": info.get("marketCap", 0),
        "Free Cash Flow": info.get("freeCashflow", 0),
        "Beta": info.get("beta", 0)
    }

df = pd.DataFrame(data).T

# RSI 시각화
fig_rsi = go.Figure()
for ticker in tickers:
    rsi = df.loc[ticker, "RSI"]
    # 색상 설정 수정
    color = px.colors.sequential.RdBu[int((rsi - 30) / 40 * 9)]  # RdBu 컬러 스케일 사용
    fig_rsi.add_trace(go.Scatter(
        x=[ticker], y=[rsi],
        mode='markers',
        marker=dict(size=50, color=color),
        name=ticker
    ))

fig_rsi.update_layout(title="RSI 비교", yaxis_title="RSI")
st.plotly_chart(fig_rsi)

# EBITDA와 Free Cash Flow 비교
fig_financials = go.Figure()
fig_financials.add_trace(go.Bar(x=tickers, y=df["EBITDA"], name="EBITDA"))
fig_financials.add_trace(go.Bar(x=tickers, y=df["Free Cash Flow"], name="Free Cash Flow"))
fig_financials.update_layout(title="EBITDA와 Free Cash Flow 비교", barmode='group')
st.plotly_chart(fig_financials)

# Total Assets, Total Capitalization, Beta 비교
fig_bubble = px.scatter(df, x="Total Assets", y="Total Capitalization", size="Beta", 
                        hover_name=df.index, size_max=60, 
                        title="Total Assets, Total Capitalization, Beta 비교")
st.plotly_chart(fig_bubble)
