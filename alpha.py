import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.fundamentaldata import FundamentalData
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
    .metric-card { padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .fundamental { background-color: #e6f7ff; }
    .technical { background-color: #fff7e6; }
    .screener { background-color: #e6ffe6; }
    .error { background-color: #ffebee; }
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
    
    # Tab 1: Profil Saham
    with tabs[0]:
        if not stock_data.empty:
            display_stock_profile(ticker, stock_data)
        else:
            st.error(f"Data untuk {ticker} tidak ditemukan")
    
    # Tab 2: Analisis Fundamental
    with tabs[1]:
        if use_alpha_vantage:
            display_fundamental_analysis(ticker, api_key)
        else:
            st.warning("Masukkan API Key untuk melihat analisis fundamental")
    
    # Tab 3: Analisis Teknikal
    with tabs[2]:
        if not stock_data.empty:
            display_technical_analysis(ticker, stock_data)
        else:
            st.error(f"Data untuk {ticker} tidak cukup untuk analisis teknikal")
    
    # Tab 4: Stock Screener
    with tabs[3]:
        display_stock_screener(api_key)

# Fungsi ambil data saham
def fetch_stock_data(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date,
            auto_adjust=True
        )
        
        # Cek jika data kosong
        if data.empty:
            st.error(f"Tidak ada data historis untuk {ticker}")
            return pd.DataFrame()
            
        return data
    except Exception as e:
        st.error(f"Error mengambil data: {str(e)}")
        return pd.DataFrame()

# Fungsi tampilkan profil saham - PERBAIKAN UTAMA
def display_stock_profile(ticker, data):
    if data.empty or len(data) < 2:
        st.error("Data tidak cukup untuk menampilkan profil saham")
        return
    
    st.subheader(f"Profil Saham: {ticker}")
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # PERBAIKAN: Konversi ke float
    last_close = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2])
    volume = float(data['Volume'].iloc[-1])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        currency = "$" if '.' in ticker else "Rp"
        st.metric("Harga Terakhir", f"{currency}{last_close:.2f}")
        
        change_pct = ((last_close - prev_close) / prev_close * 100)
        st.metric("Perubahan 1 Hari", f"{change_pct:.2f}%", delta_color="inverse")
   
    with col2:
        st.metric("Volume", f"{volume:,.0f}")
        market_cap = info.get('marketCap', 'N/A')
        if isinstance(market_cap, (int, float)):
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.2f}M")
        else:
            st.metric("Market Cap", "N/A")
    
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
        fund = FundamentalData(key=api_key, output_format='pandas')
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
        description = data.get('Description', ['N/A'])[0]
        st.write(f"**Deskripsi Perusahaan:** {description[:500]}{'...' if len(description) > 500 else ''}")
        
        # Rekomendasi valuasi
        st.subheader("Rekomendasi Valuasi")
        per_value = data.get('PERatio', ['N/A'])[0]
        if per_value != 'N/A' and per_value.replace('.','',1).isdigit():
            per = float(per_value)
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
        else:
            st.warning("Data PER tidak tersedia untuk valuasi")
    
    except ValueError as e:
        if "Invalid API call" in str(e):
            st.warning(f"Data fundamental tidak tersedia untuk {ticker} di Alpha Vantage")
            st.info("Hanya saham AS yang didukung untuk analisis fundamental")
        else:
            st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Error fetching fundamental data: {str(e)}")

# Fungsi analisis teknikal - PERBAIKAN UTAMA
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
                      yaxis_title='RSI',
                      yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)
    
    # Analisis sinyal
    st.subheader("Interpretasi Teknikal")
    last_rsi = data['RSI'].iloc[-1]
    
    # PERBAIKAN: Ambil nilai sebagai float
    close_last = float(data['Close'].iloc[-1])
    ma50_last = float(data['MA50'].iloc[-1])
    ma200_last = float(data['MA200'].iloc[-1])
    
    # Cek apakah cukup data untuk analisis
    if len(data) < 200:
        st.warning("Data historis kurang dari 200 hari, analisis teknikal mungkin tidak akurat")
    
    # PERBAIKAN: Pisahkan kondisi menjadi dua bagian
    trend = "Bullish" 
    if len(data) >= 200:
        if close_last > ma50_last and ma50_last > ma200_last:
            trend = "Bullish"
        else:
            trend = "Bearish"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card technical'><h4>Trend</h4><h2>{trend}</h2></div>", unsafe_allow_html=True)
    with col2:
        rsi_status = "Overbought (>70)" if last_rsi > 70 else "Oversold (<30)" if last_rsi < 30 else "Netral"
        st.markdown(f"<div class='metric-card technical'><h4>RSI</h4><h2>{last_rsi:.2f} - {rsi_status}</h2></div>", unsafe_allow_html=True)
    with col3:
        if len(data) >= 200:
            # PERBAIKAN: Gunakan nilai yang sudah diambil
            ma_status = "Golden Cross" if ma50_last > ma200_last else "Death Cross"
        else:
            ma_status = "Data tidak cukup"
        st.markdown(f"<div class='metric-card technical'><h4>MA Cross</h4><h2>{ma_status}</h2></div>", unsafe_allow_html=True)
    
    # Support & Resistance
    st.subheader("Support & Resistance")
    support, resistance = calculate_support_resistance(data)
    currency = "$" if '.' in ticker else "Rp"
    st.markdown(f"""
    - **Level Support Utama**: {currency}{support:.2f}
    - **Level Resistance Utama**: {currency}{resistance:.2f}
    """)

# Fungsi perhitungan RSI
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Gunakan rata-rata eksponensial
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, 1)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fungsi hitung support & resistance
def calculate_support_resistance(data, window=30):
    if len(data) < window:
        window = len(data)
        
    high = data['High'].iloc[-window:].max()
    low = data['Low'].iloc[-window:].min()
    close = data['Close'].iloc[-1]
    
    pivot = (high + low + close) / 3
    support = pivot * 2 - high
    resistance = pivot * 2 - low
    return support, resistance

# Fungsi stock screener
def display_stock_screener(api_key):
    st.subheader("Stock Screener")
    st.info("Fitur ini menggunakan data contoh. Untuk implementasi nyata, hubungkan ke API penyedia data saham.")
    
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
    
    # Contoh data saham
    stocks = pd.DataFrame({
        'Ticker': ['BBCA.JK', 'TLKM.JK', 'AAPL', 'MSFT', 'GOOGL', 'ASII.JK', 'UNVR.JK'],
        'Nama': ['Bank BCA', 'Telkom Indonesia', 'Apple Inc', 'Microsoft', 'Alphabet', 'Astra International', 'Unilever Indonesia'],
        'Sektor': ['Keuangan', 'Komunikasi', 'Teknologi', 'Teknologi', 'Teknologi', 'Otomotif', 'Konsumsi'],
        'PER': [18.5, 15.2, 28.3, 32.1, 24.8, 12.7, 30.5],
        'Dividend Yield': [2.8, 3.2, 0.6, 0.8, 0.0, 4.5, 3.8],
        'ROE': [22.3, 18.7, 40.5, 35.2, 22.8, 15.4, 25.9],
        'EPS Growth': [14.5, 12.3, 25.7, 18.4, 20.1, 8.9, 12.7],
        'Market Cap': ['Large Cap', 'Large Cap', 'Large Cap', 'Large Cap', 'Large Cap', 'Large Cap', 'Mid Cap']
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
    
    if not filtered.empty:
        # Format angka
        styled_df = filtered.style.format({
            'PER': '{:.1f}',
            'Dividend Yield': '{:.1f}%',
            'ROE': '{:.1f}%',
            'EPS Growth': '{:.1f}%'
        }).highlight_max(subset=['ROE', 'EPS Growth'], color='lightgreen')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Rekomendasi
        top_pick = filtered.iloc[0]
        st.success(f"**Top Pick:** {top_pick['Ticker']} - {top_pick['Nama']}")
        st.markdown(f"""
        - **Alasan:** 
          - PER kompetitif ({top_pick['PER']}) 
          - ROE tinggi ({top_pick['ROE']}%)
          - Dividend Yield: {top_pick['Dividend Yield']}% di atas rata-rata pasar
        """)
    else:
        st.warning("Tidak ada saham yang memenuhi kriteria filter")

if __name__ == "__main__":
    main()
