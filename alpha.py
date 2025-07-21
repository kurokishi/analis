import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.fundamentals import Fundamentals
from alpha_vantage.techindicators import TechIndicators
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Konfigurasi aplikasi
st.set_page_config(
    page_title="AnalystPro - Analisis Saham Professional",
    layout="wide",
    page_icon="📈"
)

# Fungsi utama
def main():
    st.title("📈 AnalystPro - Real-time Stock Analysis")
    st.markdown("""
    <style>
    .metric-card { padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .fundamental { background-color: #e6f7ff; }
    .technical { background-color: #fff7e6; }
    .screener { background-color: #e6ffe6; }
    </style>
    """, unsafe_allow_html=True)
    
    # Input user
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Masukkan Kode Saham (Contoh: BBCA.JK, AAPL):", "BBCA.JK").upper()
    with col2:
        api_key = st.text_input("Masukkan Alpha Vantage API Key:", type="password")
    
    # Tab utama
    tabs = st.tabs(["📊 Profil Saham", "📈 Analisis Fundamental", "📉 Analisis Teknikal", "🔍 Stock Screener"])
    
    if not api_key:
        st.warning("API Key Alpha Vantage diperlukan untuk analisis fundamental")
        use_alpha_vantage = False
    else:
        use_alpha_vantage = True
    
    # Ambil data
    stock_data = fetch_stock_data(ticker)
    if stock_data.empty:
        st.error(f"Data untuk {ticker} tidak ditemukan")
        return
    
    # Tab 1: Profil Saham
    with tabs[0]:
        display_stock_profile(ticker, stock_data)
    
    # Tab 2: Analisis Fundamental
    with tabs[1]:
        if use_alpha_vantage:
            display_fundamental_analysis(ticker, api_key)
        else:
            st.warning("Masukkan API Key untuk melihat analisis fundamental")
    
    # Tab 3: Analisis Teknikal
    with tabs[2]:
        display_technical_analysis(ticker, stock_data)
    
    # Tab 4: Stock Screener
    with tabs[3]:
        display_stock_screener(api_key)

# Fungsi ambil data saham
def fetch_stock_data(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except:
        return pd.DataFrame()

# Fungsi tampilkan profil saham
def display_stock_profile(ticker, data):
    if data.empty:
        return
    
    st.subheader(f"Profil Saham: {ticker}")
    stock = yf.Ticker(ticker)
    info = stock.info
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Harga Terakhir", f"${data['Close'].iloc[-1]:.2f}" if '.' in ticker else f"Rp{data['Close'].iloc[-1]:,.2f}")
        st.metric("Perubahan 1 Hari", f"{((data['Close'].iloc[-1] - data['Close'].iloc[-2])/data['Close'].iloc[-2]*100):.2f}%")
    
    with col2:
        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
        st.metric("Market Cap", f"${info.get('marketCap', 'N/A')}" if isinstance(info.get('marketCap'), (int, float)) else "N/A")
    
    with col3:
        st.metric("Sektor", info.get('sector', 'N/A'))
        st.metric("Industri", info.get('industry', 'N/A'))
    
    # Tampilkan chart harga
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Harga'))
    fig.update_layout(title=f"Price Chart {ticker} (1 Tahun Terakhir)",
                    yaxis_title='Harga',
                    xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# Fungsi analisis fundamental
def display_fundamental_analysis(ticker, api_key):
    st.subheader("Analisis Fundamental")
    
    try:
        # Ambil data dari Alpha Vantage
        fund = Fundamentals(key=api_key, output_format='pandas')
        data, _ = fund.get_company_overview(symbol=ticker.split('.')[0])
        
        # Tampilkan metrik utama
        cols = st.columns(4)
        metrics = [
            ('PER', 'PERatio', 'x'),
            ('PBV', 'PriceToBookRatio', 'x'),
            ('ROE', 'ReturnOnEquityTTM', '%'),
            ('Dividend Yield', 'DividendYield', '%'),
            ('EPS', 'EPS', ''),
            ('DER', 'DebtToEquity', 'x'),
            ('Profit Margin', 'GrossProfitTTM', '%'),
            ('Pertumbuhan Laba', 'QuarterlyEarningsGrowthYOY', '%')
        ]
        
        for i, (name, key, unit) in enumerate(metrics):
            value = data.get(key, ['N/A'])[0]
            if value != 'N/A' and isinstance(value, str) and value.replace('.','',1).isdigit():
                value = float(value)
                cols[i%4].metric(name, f"{value:.2f}{unit}")
            else:
                cols[i%4].metric(name, "N/A")
        
        # Analisis kualitatif
        st.subheader("Analisis Kualitatif")
        st.write(f"**Deskripsi Perusahaan:** {data.get('Description', ['N/A'])[0][:500]}...")
        
        # Rekomendasi valuasi
        st.subheader("Rekomendasi Valuasi")
        per = float(data.get('PERatio', [0])[0])
        status = "Undervalued" if per < 15 else "Overvalued" if per > 25 else "Fair Value"
        st.markdown(f"""
        <div class="metric-card fundamental">
            <h3>Valuasi: {status}</h3>
            <p>Berdasarkan PER (Current: {per:.2f}x):</p>
            <ul>
                <li>PER &lt; 15: Undervalued</li>
                <li>PER 15-25: Fair Value</li>
                <li>PER &gt; 25: Overvalued</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error fetching fundamental data: {str(e)}")

# Fungsi analisis teknikal
def display_technical_analysis(ticker, data):
    st.subheader("Analisis Teknikal")
    
    if data.empty:
        return
    
    # Hitung indikator teknikal
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_rsi(data['Close'])
    
    # Plot harga dan moving averages
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Harga', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='MA200', line=dict(color='green')))
    fig1.update_layout(title=f"Moving Averages {ticker}",
                      yaxis_title='Harga')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot RSI
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
    fig2.add_hline(y=70, line_dash="dash", line_color="red")
    fig2.add_hline(y=30, line_dash="dash", line_color="green")
    fig2.update_layout(title="Relative Strength Index (RSI)",
                      yaxis_title='RSI')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Analisis sinyal
    st.subheader("Interpretasi Teknikal")
    last_rsi = data['RSI'].iloc[-1]
    trend = "Bullish" if data['Close'].iloc[-1] > data['MA50'].iloc[-1] > data['MA200'].iloc[-1] else "Bearish"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card technical'><h4>Trend</h4><h2>{trend}</h2></div>", unsafe_allow_html=True)
    with col2:
        rsi_status = "Overbought (>70)" if last_rsi > 70 else "Oversold (<30)" if last_rsi < 30 else "Netral"
        st.markdown(f"<div class='metric-card technical'><h4>RSI</h4><h2>{last_rsi:.2f} - {rsi_status}</h2></div>", unsafe_allow_html=True)
    with col3:
        ma_status = "Golden Cross" if data['MA50'].iloc[-1] > data['MA200'].iloc[-1] else "Death Cross"
        st.markdown(f"<div class='metric-card technical'><h4>MA Cross</h4><h2>{ma_status}</h2></div>", unsafe_allow_html=True)
    
    # Support & Resistance
    st.subheader("Support & Resistance")
    support, resistance = calculate_support_resistance(data)
    st.markdown(f"""
    - **Level Support Utama**: ${support:.2f} (Saham {ticker})
    - **Level Resistance Utama**: ${resistance:.2f} (Saham {ticker})
    """)

# Fungsi perhitungan RSI
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Fungsi hitung support & resistance
def calculate_support_resistance(data):
    pivot = (data['High'].iloc[-30:].max() + data['Low'].iloc[-30:].min() + data['Close'].iloc[-1]) / 3
    support = pivot * 2 - data['High'].iloc[-30:].max()
    resistance = pivot * 2 - data['Low'].iloc[-30:].min()
    return support, resistance

# Fungsi stock screener
def display_stock_screener(api_key):
    st.subheader("Stock Screener")
    
    # Parameter screening
    with st.expander("Filter Saham"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_per = st.slider("Maksimal PER", 0.0, 50.0, 20.0)
            min_div = st.slider("Minimal Dividend Yield (%)", 0.0, 10.0, 2.0)
        with col2:
            min_roe = st.slider("Minimal ROE (%)", 0.0, 50.0, 15.0)
            min_eps_growth = st.slider("Minimal EPS Growth (%)", -20.0, 100.0, 10.0)
        with col3:
            sector = st.selectbox("Sektor", ["Semua", "Teknologi", "Keuangan", "Kesehatan", "Konsumsi"])
            market_cap = st.selectbox("Market Cap", ["Semua", "Large Cap", "Mid Cap", "Small Cap"])
    
    # Contoh data saham (dalam real app, ini akan diambil dari API)
    stocks = pd.DataFrame({
        'Ticker': ['BBCA.JK', 'TLKM.JK', 'AAPL', 'MSFT', 'GOOGL'],
        'Nama': ['Bank BCA', 'Telkom Indonesia', 'Apple Inc', 'Microsoft', 'Alphabet'],
        'Sektor': ['Keuangan', 'Komunikasi', 'Teknologi', 'Teknologi', 'Teknologi'],
        'PER': [18.5, 15.2, 28.3, 32.1, 24.8],
        'Dividend Yield': [2.8, 3.2, 0.6, 0.8, 0.0],
        'ROE': [22.3, 18.7, 40.5, 35.2, 22.8],
        'EPS Growth': [14.5, 12.3, 25.7, 18.4, 20.1],
        'Market Cap': ['Large Cap', 'Large Cap', 'Large Cap', 'Large Cap', 'Large Cap']
    })
    
    # Filter saham
    filtered = stocks[
        (stocks['PER'] <= max_per) &
        (stocks['Dividend Yield'] >= min_div) &
        (stocks['ROE'] >= min_roe) &
        (stocks['EPS Growth'] >= min_eps_growth)
    ]
    
    if sector != "Semua":
        filtered = filtered[filtered['Sektor'] == sector]
    
    if market_cap != "Semua":
        filtered = filtered[filtered['Market Cap'] == market_cap]
    
    # Tampilkan hasil
    st.subheader(f"Saham yang Memenuhi Kriteria: {len(filtered)}")
    st.dataframe(filtered.style.highlight_max(subset=['ROE', 'EPS Growth'], color='lightgreen'), use_container_width=True)
    
    # Rekomendasi
    if not filtered.empty:
        st.success(f"**Top Pick:** {filtered.iloc[0]['Ticker']} - {filtered.iloc[0]['Nama']}")
        st.markdown(f"""
        - **Alasan:** PER kompetitif ({filtered.iloc[0]['PER']}) dengan ROE tinggi ({filtered.iloc[0]['ROE']}%)
        - **Dividend Yield:** {filtered.iloc[0]['Dividend Yield']}% di atas rata-rata pasar
        """)

if __name__ == "__main__":
    main()
