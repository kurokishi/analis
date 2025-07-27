import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

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
    .api-option { margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Input user
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Masukkan Kode Saham (Contoh: BBCA.JK, AAPL):", "AAPL").upper()
    with col2:
        api_source = st.selectbox("Pilih Sumber Data Fundamental:", 
                                ["Yahoo Finance", "Financial Modeling Prep"])
        
        if api_source == "Financial Modeling Prep":
            api_key = st.text_input("Masukkan Financial Modeling Prep API Key:", type="password")
        else:
            api_key = None
    
    # Tab utama
    tabs = st.tabs(["📊 Profil Saham", "📈 Analisis Fundamental", "📉 Analisis Teknikal", "🔍 Stock Screener"])
    
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
        display_fundamental_analysis(ticker, api_source, api_key)
    
    # Tab 3: Analisis Teknikal
    with tabs[2]:
        if not stock_data.empty:
            display_technical_analysis(ticker, stock_data)
        else:
            st.error(f"Data untuk {ticker} tidak cukup untuk analisis teknikal")
    
    # Tab 4: Stock Screener
    with tabs[3]:
        display_stock_screener(api_source, api_key)

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

# Fungsi tampilkan profil saham
def display_stock_profile(ticker, data):
    if data.empty or len(data) < 2:
        st.error("Data tidak cukup untuk menampilkan profil saham")
        return
    
    st.subheader(f"Profil Saham: {ticker}")
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Konversi ke float dengan cara yang benar
    last_close = float(data['Close'].iloc[-1]) if not data.empty else 0
    prev_close = float(data['Close'].iloc[-2]) if len(data) >= 2 else 0
    volume = float(data['Volume'].iloc[-1]) if not data.empty else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        currency = "$" if '.' in ticker else "Rp"
        st.metric("Harga Terakhir", f"{currency}{last_close:.2f}")
        
        change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0
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

# Fungsi analisis fundamental yang mendukung multiple API
def display_fundamental_analysis(ticker, api_source, api_key=None):
    st.subheader("Analisis Fundamental")
    
    if api_source == "Financial Modeling Prep" and not api_key:
        st.warning("API Key diperlukan untuk Financial Modeling Prep")
        return
    
    try:
        if api_source == "Yahoo Finance":
            display_yfinance_fundamental(ticker)
        elif api_source == "Financial Modeling Prep":
            display_fmp_fundamental(ticker, api_key)
        else:
            st.warning("Sumber data tidak didukung")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

# Fungsi fundamental dengan Yahoo Finance
def display_yfinance_fundamental(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Tampilkan metrik utama dari Yahoo Finance
        cols = st.columns(4)
        metrics = [
            ('PER', 'trailingPE', 'x'),
            ('PBV', 'priceToBook', 'x'),
            ('ROE', 'returnOnEquity', '%'),
            ('Dividend Yield', 'dividendYield', '%'),
            ('EPS', 'trailingEps', ''),
            ('DER', 'debtToEquity', 'x'),
            ('Profit Margin', 'profitMargins', '%'),
            ('Pertumbuhan Laba', 'earningsQuarterlyGrowth', '%')
        ]
        
        for i, (name, key, unit) in enumerate(metrics):
            value = info.get(key, 'N/A')
            if value != 'N/A' and isinstance(value, (int, float)):
                cols[i%4].metric(name, f"{value:.2f}{unit}")
            else:
                cols[i%4].metric(name, "N/A")
        
        # Analisis kualitatif
        st.subheader("Analisis Kualitatif")
        description = info.get('longBusinessSummary', 'N/A')
        st.write(f"**Deskripsi Perusahaan:** {description[:500]}{'...' if len(description) > 500 else ''}")
        
        # Rekomendasi valuasi
        st.subheader("Rekomendasi Valuasi")
        per_value = info.get('trailingPE', 'N/A')
        if per_value != 'N/A' and isinstance(per_value, (int, float)):
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
            
        st.info("Sumber data: Yahoo Finance (gratis, tanpa API Key)")
    
    except Exception as e:
        st.error(f"Error mengambil data dari Yahoo Finance: {str(e)}")

# Fungsi fundamental dengan Financial Modeling Prep
def display_fmp_fundamental(ticker, api_key):
    try:
        # Bersihkan ticker untuk FMP (tanpa suffix pasar)
        clean_ticker = ticker.split('.')[0]
        
        # Ambil data profil perusahaan
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{clean_ticker}?apikey={api_key}"
        profile_response = requests.get(profile_url)
        profile_data = profile_response.json()
        
        if not profile_data or 'Error' in profile_data:
            st.error(f"Tidak dapat mengambil data profil untuk {ticker} dari Financial Modeling Prep")
            return
        
        profile = profile_data[0] if isinstance(profile_data, list) else profile_data
        
        # Ambil data rasio keuangan
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{clean_ticker}?apikey={api_key}"
        ratios_response = requests.get(ratios_url)
        ratios_data = ratios_response.json()
        
        if not ratios_data or 'Error' in ratios_data:
            st.warning(f"Tidak dapat mengambil data rasio untuk {ticker}")
            ratios = {}
        else:
            ratios = ratios_data[0] if isinstance(ratios_data, list) else ratios_data
        
        # Tampilkan metrik utama
        cols = st.columns(4)
        
        # Siapkan data metrik
        metrics = [
            ('PER', 'pe', 'x', profile.get('pe', 'N/A')),
            ('PBV', 'pb', 'x', profile.get('pb', 'N/A')),
            ('ROE', 'returnOnEquityTTM', '%', ratios.get('returnOnEquityTTM', 'N/A')),
            ('Dividend Yield', 'dividendYield', '%', profile.get('lastDiv', 'N/A')),
            ('EPS', 'eps', '', profile.get('eps', 'N/A')),
            ('DER', 'debtEquityRatio', 'x', profile.get('debtEquityRatio', 'N/A')),
            ('Profit Margin', 'profitMarginTTM', '%', ratios.get('profitMarginTTM', 'N/A')),
            ('Pertumbuhan Laba', 'earningsGrowth', '%', profile.get('earningsGrowth', 'N/A'))
        ]
        
        for i, (name, key, unit, value) in enumerate(metrics):
            if value != 'N/A' and isinstance(value, (int, float)):
                cols[i%4].metric(name, f"{value:.2f}{unit}")
            else:
                cols[i%4].metric(name, "N/A")
        
        # Analisis kualitatif
        st.subheader("Analisis Kualitatif")
        description = profile.get('description', 'N/A')
        st.write(f"**Deskripsi Perusahaan:** {description[:500]}{'...' if len(description) > 500 else ''}")
        
        # Rekomendasi valuasi
        st.subheader("Rekomendasi Valuasi")
        per_value = profile.get('pe', 'N/A')
        if per_value != 'N/A' and isinstance(per_value, (int, float)):
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
            
        st.info("Sumber data: Financial Modeling Prep")
    
    except Exception as e:
        st.error(f"Error mengambil data dari Financial Modeling Prep: {str(e)}")

# Fungsi analisis teknikal
def display_technical_analysis(ticker, data):
    st.subheader("Analisis Teknikal")
    
    if data.empty or len(data) < 50:  # Minimal 50 data untuk MA50
        st.error(f"Data untuk {ticker} tidak cukup untuk analisis teknikal")
        return
    
    # Hitung indikator teknikal
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean() if len(data) >= 200 else np.nan
    data['RSI'] = compute_rsi(data['Close'])
    
    # Plot harga dan moving averages
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Harga', line=dict(color='blue')))
    
    # Hanya tambahkan MA jika ada cukup data
    if len(data) >= 50:
        fig1.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='orange')))
    if len(data) >= 200:
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
    
    # Ambil nilai sebagai float dengan cara yang benar
    last_rsi = float(data['RSI'].iloc[-1]) if not data.empty else 0
    close_last = float(data['Close'].iloc[-1]) if not data.empty else 0
    ma50_last = float(data['MA50'].iloc[-1]) if not data.empty and not pd.isna(data['MA50'].iloc[-1]) else 0
    ma200_last = float(data['MA200'].iloc[-1]) if not data.empty and not pd.isna(data['MA200'].iloc[-1]) else 0
    
    # Cek apakah cukup data untuk analisis
    if len(data) < 200:
        st.warning("Data historis kurang dari 200 hari, analisis teknikal mungkin tidak akurat")
    
    # Pisahkan kondisi menjadi dua bagian
    trend = "Bullish" 
    if len(data) >= 200 and not pd.isna(ma50_last) and not pd.isna(ma200_last):
        if close_last > ma50_last and ma50_last > ma200_last:
            trend = "Bullish"
        else:
            trend = "Bearish"
    elif len(data) >= 50 and not pd.isna(ma50_last):
        trend = "Bullish" if close_last > ma50_last else "Bearish"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card technical'><h4>Trend</h4><h2>{trend}</h2></div>", unsafe_allow_html=True)
    with col2:
        rsi_status = "Overbought (>70)" if last_rsi > 70 else "Oversold (<30)" if last_rsi < 30 else "Netral"
        st.markdown(f"<div class='metric-card technical'><h4>RSI</h4><h2>{last_rsi:.2f} - {rsi_status}</h2></div>", unsafe_allow_html=True)
    with col3:
        if len(data) >= 200 and not pd.isna(ma50_last) and not pd.isna(ma200_last):
            ma_status = "Golden Cross" if ma50_last > ma200_last else "Death Cross"
            st.markdown(f"<div class='metric-card technical'><h4>MA Cross</h4><h2>{ma_status}</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card technical'><h4>MA Cross</h4><h2>Data tidak cukup</h2></div>", unsafe_allow_html=True)
    
    # Support & Resistance
    st.subheader("Support & Resistance")
    support, resistance = calculate_support_resistance(data)
    currency = "$" if '.' in ticker else "Rp"
    
    # Pastikan nilai float
    try:
        support_val = float(support)
        resistance_val = float(resistance)
        st.markdown(f"""
        - **Level Support Utama**: {currency}{support_val:.2f}
        - **Level Resistance Utama**: {currency}{resistance_val:.2f}
        """)
    except (TypeError, ValueError):
        st.error("Tidak dapat menghitung support dan resistance")

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
    
    # Ambil nilai float secara eksplisit
    try:
        high = float(data['High'].iloc[-window:].max())
        low = float(data['Low'].iloc[-window:].min())
        close = float(data['Close'].iloc[-1])
        
        pivot = (high + low + close) / 3
        support = pivot * 2 - high
        resistance = pivot * 2 - low
        return support, resistance
    except Exception as e:
        print(f"Error calculating support/resistance: {e}")
        return 0, 0

# Fungsi stock screener
def display_stock_screener(api_source, api_key):
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
