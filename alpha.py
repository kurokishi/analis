import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time

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
    .stocks-table { font-size: 0.9rem; }
    .positive { color: green; }
    .negative { color: red; }
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
            st.session_state.fmp_api_key = api_key
        else:
            api_key = None
    
    # Tab utama - semua tab didefinisikan sekaligus
    tabs = st.tabs([
        "📊 Profil Saham", 
        "📈 Analisis Fundamental", 
        "📉 Analisis Teknikal", 
        "🔍 Stock Screener",
        "📍 Rekomendasi Time Horizon"
    ])
    
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
        
    # Tab 5: Rekomendasi Time Horizon
    with tabs[4]:
        display_investment_recommendations()

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
    
    # Ambil nilai langsung tanpa konversi float
    last_close = data['Close'].iloc[-1] if not data.empty else 0
    prev_close = data['Close'].iloc[-2] if len(data) >= 2 else 0
    volume = data['Volume'].iloc[-1] if not data.empty else 0
    
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
    
    # Ambil nilai langsung tanpa konversi float
    last_rsi = data['RSI'].iloc[-1] if not data.empty else 0
    close_last = data['Close'].iloc[-1] if not data.empty else 0
    ma50_last = data['MA50'].iloc[-1] if not data.empty and not pd.isna(data['MA50'].iloc[-1]) else 0
    ma200_last = data['MA200'].iloc[-1] if not data.empty and not pd.isna(data['MA200'].iloc[-1]) else 0
    
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
        support_val = support
        resistance_val = resistance
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
    
    try:
        # Ambil nilai langsung
        high = data['High'].iloc[-window:].max()
        low = data['Low'].iloc[-window:].min()
        close = data['Close'].iloc[-1]
        
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

def display_investment_recommendations():
    st.header("📍 Rekomendasi Saham Berdasarkan Time Horizon")
    st.info("Rekomendasi saham berdasarkan horizon waktu dan profil risiko Anda menggunakan data real-time")
    
    # Input pengguna
    col1, col2 = st.columns(2)
    with col1:
        risk_profile = st.selectbox("Pilih Profil Risiko Anda:", ["Rendah", "Sedang", "Tinggi"])
    with col2:
        market = st.selectbox("Pilih Pasar Saham:", ["AS (NYSE/NASDAQ)", "Indonesia (IDX)"])
    
    # Daftar saham untuk masing-masing pasar
    if market == "AS (NYSE/NASDAQ)":
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'JNJ', 'PG', 'DIS']
        market_suffix = ""
    else:
        tickers = ['BBCA.JK', 'TLKM.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'ASII.JK', 'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'CPIN.JK']
        market_suffix = ".JK"
    
    # Cek apakah API key tersedia
    if 'fmp_api_key' in st.session_state and st.session_state.fmp_api_key:
        api_key = st.session_state.fmp_api_key
        use_real_data = True
    else:
        st.warning("API Key Financial Modeling Prep tidak ditemukan. Menggunakan data contoh.")
        use_real_data = False
    
    # Ambil data saham
    if use_real_data:
        with st.spinner('Mengambil data real-time...'):
            df = fetch_real_time_stock_data(tickers, market, api_key)
    else:
        df = generate_sample_data(tickers, market)
    
    # Tampilkan data
    if df is not None and not df.empty:
        # Filter berdasarkan profil risiko
        if risk_profile == "Rendah":
            filtered_df = df[(df['Dividend Yield'] > 2.0) & (df['PER'] < 25)]
        elif risk_profile == "Sedang":
            filtered_df = df[(df['ROE'] > 15) & (df['EPS Growth'] > 10)]
        elif risk_profile == "Tinggi":
            filtered_df = df[df['EPS Growth'] > 20]
        else:
            filtered_df = df
        
        # Segmentasi berdasarkan horizon investasi
        st.subheader("🌱 Jangka Pendek (< 3 bulan)")
        st.markdown("**Kriteria:** Pertumbuhan laba tinggi (EPS Growth > 10%) dan valuasi wajar (PER < 25)")
        short_term = filtered_df[(filtered_df['EPS Growth'] > 10) & (filtered_df['PER'] < 25)]
        display_recommendation_table(short_term, ['Ticker', 'Nama', 'PER', 'EPS Growth', 'Sektor'])
        
        st.subheader("⏳ Jangka Menengah (3-12 bulan)")
        st.markdown("**Kriteria:** Profitabilitas tinggi (ROE > 15%) dan imbal hasil dividen baik (Dividend Yield > 2%)")
        mid_term = filtered_df[(filtered_df['ROE'] > 15) & (filtered_df['Dividend Yield'] > 2)]
        display_recommendation_table(mid_term, ['Ticker', 'Nama', 'ROE', 'Dividend Yield', 'Sektor'])
        
        st.subheader("🌳 Jangka Panjang (> 1 tahun)")
        st.markdown("**Kriteria:** Margin laba sehat (Profit Margin > 10%) dan valuasi aset wajar (PBV < 5)")
        long_term = filtered_df[(filtered_df['Profit Margin'] > 10) & (filtered_df['PBV'] < 5)]
        display_recommendation_table(long_term, ['Ticker', 'Nama', 'Profit Margin', 'PBV', 'Sektor'])
        
        st.caption(f"Sumber data: {'Financial Modeling Prep API' if use_real_data else 'Data Contoh'} - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    elif df is not None and df.empty:
        st.warning("Tidak ada data saham yang ditemukan")
    else:
        st.error("Gagal mengambil data saham. Silakan coba lagi atau gunakan API Key yang valid.")

def fetch_real_time_stock_data(tickers, market, api_key):
    """Ambil data real-time dari Financial Modeling Prep API"""
    data = []
    
    for ticker in tickers:
        try:
            # Bersihkan ticker untuk API
            clean_ticker = ticker.replace('.JK', '') if market == "Indonesia (IDX)" else ticker
            
            # Ambil data profil perusahaan
            profile_url = f"https://financialmodelingprep.com/api/v3/profile/{clean_ticker}?apikey={api_key}"
            profile_response = requests.get(profile_url)
            profile_data = profile_response.json()
            
            if not profile_data or 'Error' in profile_data:
                st.warning(f"Data tidak ditemukan untuk {ticker}")
                continue
            
            profile = profile_data[0] if isinstance(profile_data, list) else profile_data
            
            # Ambil data rasio keuangan
            ratios_url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{clean_ticker}?apikey={api_key}"
            ratios_response = requests.get(ratios_url)
            ratios_data = ratios_response.json()
            
            if not ratios_data or 'Error' in ratios_data:
                st.warning(f"Rasio keuangan tidak ditemukan untuk {ticker}")
                ratios = {}
            else:
                ratios = ratios_data[0] if isinstance(ratios_data, list) else ratios_data
            
            # Ambil data pertumbuhan laba (contoh: 5 tahun terakhir)
            growth_url = f"https://financialmodelingprep.com/api/v3/income-statement-growth/{clean_ticker}?limit=5&apikey={api_key}"
            growth_response = requests.get(growth_url)
            growth_data = growth_response.json()
            
            eps_growth = 0
            if growth_data and isinstance(growth_data, list) and len(growth_data) > 0:
                # Hitung rata-rata pertumbuhan EPS
                eps_growth_values = [item.get('growthEps', 0) for item in growth_data if item.get('growthEps') is not None]
                eps_growth = sum(eps_growth_values) / len(eps_growth_values) * 100 if eps_growth_values else 0
            
            # Simpan data
            data.append({
                'Ticker': ticker,
                'Nama': profile.get('companyName', ticker),
                'PER': profile.get('pe', 0),
                'PBV': profile.get('pb', 0),
                'ROE': ratios.get('returnOnEquityTTM', 0) * 100 if ratios.get('returnOnEquityTTM') else 0,
                'Dividend Yield': profile.get('lastDiv', 0) / profile.get('price', 1) * 100 if profile.get('lastDiv') and profile.get('price') else 0,
                'EPS Growth': eps_growth,
                'Profit Margin': ratios.get('profitMarginTTM', 0) * 100 if ratios.get('profitMarginTTM') else 0,
                'Sektor': profile.get('sector', 'N/A')
            })
            
            # Jeda untuk menghindari rate limit
            time.sleep(0.2)
            
        except Exception as e:
            st.error(f"Error mengambil data untuk {ticker}: {str(e)}")
            continue
    
    if not data:
        return None
    
    return pd.DataFrame(data)

def generate_sample_data(tickers, market):
    """Hasilkan data contoh jika API tidak tersedia"""
    if market == "AS (NYSE/NASDAQ)":
        names = ['Apple Inc', 'Microsoft', 'Alphabet', 'Amazon', 'Meta', 'Tesla', 'JPMorgan', 'Johnson & Johnson', 'Procter & Gamble', 'Disney']
        sectors = ['Teknologi', 'Teknologi', 'Teknologi', 'E-commerce', 'Teknologi', 'Otomotif', 'Keuangan', 'Kesehatan', 'Konsumsi', 'Hiburan']
    else:
        names = ['Bank BCA', 'Telkom', 'Bank BRI', 'Bank BNI', 'Bank Mandiri', 'Astra International', 'Unilever', 'Indofood', 'Chicken', 'Charoen']
        sectors = ['Keuangan', 'Komunikasi', 'Keuangan', 'Keuangan', 'Keuangan', 'Otomotif', 'Konsumsi', 'Konsumsi', 'Peternakan', 'Peternakan']
    
    # Buat data secara acak dengan distribusi yang masuk akal
    np.random.seed(42)
    size = len(tickers)
    
    return pd.DataFrame({
        'Ticker': tickers,
        'Nama': names,
        'PER': np.random.uniform(5, 40, size).round(1),
        'PBV': np.random.uniform(0.5, 10, size).round(1),
        'ROE': np.random.uniform(5, 40, size).round(1),
        'Dividend Yield': np.random.uniform(0, 8, size).round(1),
        'EPS Growth': np.random.uniform(-10, 50, size).round(1),
        'Profit Margin': np.random.uniform(5, 35, size).round(1),
        'Sektor': sectors
    })

def display_recommendation_table(df, columns):
    """Tampilkan tabel rekomendasi dengan styling"""
    if not df.empty:
        # Urutkan berdasarkan kinerja
        sort_column = columns[2]  # Kolom ketiga biasanya metrik utama
        df = df.sort_values(by=sort_column, ascending=False)
        
        # Buat salinan DataFrame untuk styling
        styled_df = df[columns].copy()
        
        # Format kolom numerik
        if 'ROE' in styled_df.columns:
            styled_df['ROE'] = styled_df['ROE'].apply(lambda x: f"{x:.1f}%")
        if 'Dividend Yield' in styled_df.columns:
            styled_df['Dividend Yield'] = styled_df['Dividend Yield'].apply(lambda x: f"{x:.1f}%")
        if 'EPS Growth' in styled_df.columns:
            styled_df['EPS Growth'] = styled_df['EPS Growth'].apply(lambda x: f"{x:.1f}%")
        if 'Profit Margin' in styled_df.columns:
            styled_df['Profit Margin'] = styled_df['Profit Margin'].apply(lambda x: f"{x:.1f}%")
        if 'PER' in styled_df.columns:
            styled_df['PER'] = styled_df['PER'].apply(lambda x: f"{x:.1f}")
        if 'PBV' in styled_df.columns:
            styled_df['PBV'] = styled_df['PBV'].apply(lambda x: f"{x:.1f}")
        
        # Tampilkan tabel
        st.dataframe(styled_df, use_container_width=True)
        
        # Rekomendasi teratas
        top_pick = df.iloc[0]
        st.success(f"**Top Pick:** {top_pick['Ticker']} - {top_pick['Nama']}")
        
        # Analisis singkat
        analysis = f"""
        - **Sektor:** {top_pick['Sektor']}
        - **PER:** {top_pick.get('PER', 'N/A'):.1f}
        - **ROE:** {top_pick.get('ROE', 'N/A'):.1f}%
        - **Dividend Yield:** {top_pick.get('Dividend Yield', 'N/A'):.1f}%
        - **Pertumbuhan EPS:** {top_pick.get('EPS Growth', 'N/A'):.1f}%
        """
        st.markdown(analysis)
    else:
        st.warning("Tidak ada saham yang memenuhi kriteria untuk horizon waktu ini")

if __name__ == "__main__":
    # Simpan API key di session state jika ada
    if 'fmp_api_key' not in st.session_state:
        st.session_state.fmp_api_key = None
    
    main()
