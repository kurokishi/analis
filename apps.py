import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random
import base64
import time 

# Perbaikan impor NewsApiClient
try:
    from newsapi.newsapi_client import NewsApiClient
except ImportError:
    NewsApiClient = None
    st.warning("NewsAPI client tidak tersedia. Fitur berita akan dibatasi.")

# Perbaikan untuk vaderSentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    import subprocess
    import sys
    import warnings
    warnings.filterwarnings("ignore", message=".*vaderSentiment.*")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"])
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except:
        SentimentIntensityAnalyzer = None
        st.warning("Modul vaderSentiment tidak tersedia. Analisis sentimen akan dibatasi.")

from textblob import TextBlob
import feedparser

# Konfigurasi halaman untuk PWA
st.set_page_config(
    page_title="Stock Analysis Toolkit Pro+",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"  # Diubah untuk responsif mobile
)

# ==========================================
# FUNGSI UNTUK PWA (PROGRESSIVE WEB APP)
# ==========================================

def add_pwa_meta():
    """Menambahkan meta tags untuk PWA"""
    pwa_meta = """
        <link rel="manifest" href="manifest.json">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="application-name" content="Stock Analysis">
        <meta name="apple-mobile-web-app-title" content="Stock Analysis">
        <meta name="theme-color" content="#1e3a8a">
        <meta name="msapplication-navbutton-color" content="#1e3a8a">
        <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1, maximum-scale=1, viewport-fit=cover">
        <link rel="apple-touch-icon" href="icon-192x192.png">
    """
    st.markdown(pwa_meta, unsafe_allow_html=True)

def create_manifest():
    """Membuat file manifest.json untuk PWA"""
    manifest = {
        "name": "Stock Analysis Toolkit Pro+",
        "short_name": "Stock Analysis",
        "start_url": ".",
        "display": "standalone",
        "theme_color": "#1e3a8a",
        "background_color": "#ffffff",
        "icons": [
            {
                "src": "icon-192x192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "icon-512x512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    return manifest

def generate_icon_base64():
    """Generate base64 encoded icon untuk demo (dalam aplikasi nyata gunakan file eksternal)"""
    # Icon placeholder (dalam aplikasi nyata gunakan file eksternal)
    icon_data = base64.b64encode(b"").decode("utf-8")
    return icon_data

def setup_pwa():
    """Setup untuk PWA"""
    add_pwa_meta()
    
    # Generate manifest (dalam aplikasi nyata simpan sebagai file eksternal)
    manifest = create_manifest()
    st.session_state.manifest = manifest

# Panggil setup PWA
setup_pwa()

# ==========================================
# CSS RESPONSIF UNTUK MOBILE
# ==========================================

def inject_responsive_css():
    """Menyuntikkan CSS responsif untuk perangkat mobile"""
    mobile_css = """
    <style>
        /* Responsive sidebar */
        @media (max-width: 768px) {
            .sidebar .sidebar-content {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            div[data-testid="stSidebarUserContent"] {
                padding: 1rem 0.5rem !important;
            }
            
            div[data-testid="stVerticalBlock"] > div {
                padding: 0.5rem !important;
            }
            
            /* Menu lebih kompak */
            .sidebar .stSelectbox, .sidebar .stTextInput, .sidebar .stButton {
                margin-bottom: 0.5rem !important;
            }
            
            /* Header lebih kecil */
            .main h1 {
                font-size: 1.5rem !important;
            }
            
            /* Kolom metrik diubah menjadi vertikal */
            .stMetric {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Grafik lebih kecil */
            .stPlotlyChart {
                height: 300px !important;
            }
            
            /* Hide complex elements on mobile */
            .mobile-hide {
                display: none !important;
            }
        }
        
        /* Tablet view */
        @media (min-width: 769px) and (max-width: 1024px) {
            .stMetric {
                width: 50% !important;
            }
        }
        
        /* Tombol install PWA */
        #installButton {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #1e3a8a;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 1000;
            display: none; /* Sembunyikan secara default */
        }
        
        /* Tampilan mobile-friendly untuk semua ukuran */
        .mobile-optimized {
            padding: 0.5rem !important;
        }
        
        .mobile-full-width {
            width: 100% !important;
        }
        
        .mobile-collapsible {
            margin-bottom: 0.5rem !important;
        }
        
        /* Responsif untuk tabel */
        .dataframe {
            font-size: 0.8rem !important;
        }
    </style>
    """
    st.markdown(mobile_css, unsafe_allow_html=True)

inject_responsive_css()

# ==========================================
# LOGIKA INSTALASI PWA
# ==========================================

def add_pwa_install_button():
    """Menambahkan tombol instalasi PWA"""
    pwa_install_script = """
    <script>
        let deferredPrompt;
        const installButton = document.createElement('button');
        installButton.id = 'installButton';
        installButton.innerHTML = '📱';
        installButton.title = 'Install App';
        document.body.appendChild(installButton);
        
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            installButton.style.display = 'block';
        });
        
        installButton.addEventListener('click', async () => {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                const { outcome } = await deferredPrompt.userChoice;
                console.log(`User response to the install prompt: ${outcome}`);
                deferredPrompt = null;
                installButton.style.display = 'none';
            }
        });
        
        window.addEventListener('appinstalled', () => {
            console.log('PWA was installed');
            installButton.style.display = 'none';
            deferredPrompt = null;
        });
    </script>
    """
    st.markdown(pwa_install_script, unsafe_allow_html=True)

add_pwa_install_button()

# ==========================================
# FUNGSI UTAMA (SAMA SEBELUMNYA)
# ==========================================

# Fungsi untuk mendapatkan API key dengan key unik
def get_fmp_api_key():
    if 'fmp_api_key' not in st.session_state:
        with st.sidebar:
            st.subheader("FinancialModelingPrep API")
            # PERBAIKAN: Tambahkan key unik
            api_key = st.text_input("Masukkan API Key FMP", type="password", key="fmp_api_key_input")
            if st.button("Simpan API Key", key="save_fmp_api_key"):
                st.session_state.fmp_api_key = api_key
                st.success("API Key disimpan!")
                st.rerun()
        return None
    return st.session_state.fmp_api_key

# Fungsi untuk mendapatkan NewsAPI key dengan key unik
def get_news_api_key():
    if 'news_api_key' not in st.session_state:
        with st.sidebar:
            st.subheader("NewsAPI Configuration")
            # PERBAIKAN: Tambahkan key unik
            api_key = st.text_input("Masukkan NewsAPI Key", type="password", key="news_api_key_input")
            if st.button("Simpan NewsAPI Key", key="save_news_api_key"):
                st.session_state.news_api_key = api_key
                st.success("NewsAPI Key disimpan!")
                st.rerun()
        return None
    return st.session_state.news_api_key

# Fungsi untuk mendapatkan data real-time
def get_realtime_data(ticker):
    try:
        # Tambahkan penundaan untuk menghindari rate limit
        time.sleep(0.5)  # Delay 500ms antara permintaan
        
        stock = yf.Ticker(ticker)
        
        # Coba dapatkan data real-time
        try:
            hist = stock.history(period="1d", interval="5m")
        except Exception as e:
            # Fallback ke data harian jika intraday gagal
            hist = stock.history(period="1d", interval="1d")
        
        if hist.empty:
            # Coba gunakan data historis terbaru sebagai fallback
            hist = stock.history(period="1d")
            if hist.empty:
                return None, None, None, None
        
        last_price = hist['Close'].iloc[-1]
        
        # Dapatkan harga penutupan sebelumnya
        try:
            prev_close = stock.info.get('previousClose', last_price)
        except:
            # Jika gagal, gunakan harga pertama hari ini
            prev_close = hist['Open'].iloc[0] if not hist.empty else last_price
        
        change = last_price - prev_close
        change_percent = (change / prev_close) * 100
        
        return last_price, change, change_percent, hist
        
    except Exception as e:
        # Tangani error khusus rate limit
        if "Too Many Requests" in str(e) or "429" in str(e):
            st.warning("Yahoo Finance rate limit terlampaui. Data mungkin tidak real-time.")
            # Coba dapatkan data historis sebagai fallback
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[0]
                    change = last_price - prev_close
                    change_percent = (change / prev_close) * 100
                    return last_price, change, change_percent, hist
            except:
                pass
        
        st.error(f"Error fetching data: {str(e)}")
        return None, None, None, None

# Fungsi untuk memproses file upload
def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Harap upload file CSV atau Excel.")
            return pd.DataFrame()
        
        # Konversi kolom harga ke numeric
        if 'Avg Price' in df.columns:
            df['Avg Price'] = df['Avg Price'].replace('[Rp, ]', '', regex=True).astype(float)
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()

# Fungsi untuk mendapatkan data fundamental dari FMP
def get_fmp_data(ticker, api_key):
    try:
        # Dapatkan profil perusahaan
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
        profile_response = requests.get(profile_url)
        profile_data = profile_response.json()
        
        if not profile_data:
            st.warning(f"Data perusahaan tidak ditemukan untuk {ticker}")
            return None
        
        # Dapatkan rasio keuangan
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=annual&apikey={api_key}"
        ratios_response = requests.get(ratios_url)
        ratios_data = ratios_response.json()
        
        # Dapatkan laporan arus kas
        cashflow_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=annual&apikey={api_key}"
        cashflow_response = requests.get(cashflow_url)
        cashflow_data = cashflow_response.json()
        
        # Dapatkan data pasar
        quote_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
        quote_response = requests.get(quote_url)
        quote_data = quote_response.json()
        
        # Dapatkan pertumbuhan pendapatan
        growth_url = f"https://financialmodelingprep.com/api/v3/income-statement-growth/{ticker}?period=annual&apikey={api_key}"
        growth_response = requests.get(growth_url)
        growth_data = growth_response.json()
        
        # Gabungkan semua data
        fmp_data = {
            'profile': profile_data[0] if profile_data else {},
            'ratios': ratios_data[0] if ratios_data else {},
            'cashflow': cashflow_data[0] if cashflow_data else {},
            'quote': quote_data[0] if quote_data else {},
            'growth': growth_data[0] if growth_data else {}
        }
        
        return fmp_data
    except Exception as e:
        st.error(f"Error fetching FMP data: {str(e)}")
        return None

# Fungsi untuk memperbarui data real-time portfolio
def update_portfolio_data(portfolio_df):
    if portfolio_df.empty:
        return portfolio_df
    
    df = portfolio_df.copy()
    lot_balance_col = 'Lot Balance'
    
    # Dapatkan harga real-time
    current_prices = []
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        
        # Gunakan cache jika tersedia
        cache_key = f"price_{ticker}"
        if cache_key in st.session_state:
            last_price = st.session_state[cache_key]
        else:
            last_price, _, _, _ = get_realtime_data(ticker)
            # Simpan dalam cache selama 60 detik
            st.session_state[cache_key] = last_price
            st.session_state[f"{cache_key}_time"] = time.time()
        
        # Jika data real-time gagal, gunakan harga rata-rata sebagai fallback
        if last_price is None:
            last_price = row['Avg Price']
        
        current_prices.append(last_price)
    
    df['Current Price'] = current_prices
    df['Current Value'] = df[lot_balance_col] * df['Current Price']
    df['Profit/Loss'] = df['Current Value'] - (df[lot_balance_col] * df['Avg Price'])
    df['Profit/Loss %'] = (df['Current Value'] / (df[lot_balance_col] * df['Avg Price']) - 1) * 100
    
    return df

# Tambahkan fungsi untuk membersihkan cache
def clear_price_cache():
    # Hapus cache yang lebih lama dari 60 detik
    current_time = time.time()
    keys_to_delete = []
    for key in st.session_state.keys():
        if key.startswith("price_") and f"{key}_time" in st.session_state:
            if current_time - st.session_state[f"{key}_time"] > 60:  # 60 detik
                keys_to_delete.append(key)
                keys_to_delete.append(f"{key}_time")
    
    for key in keys_to_delete:
        del st.session_state[key]

# Fungsi analisis DCA dengan data real-time - DIUBAH UNTUK RESPONSIF
def dca_analysis(df):
    if df.empty:
        return df
    
    # Perbarui data dengan harga real-time
    df = update_portfolio_data(df)
    
    st.subheader("📊 Analisis Dollar Cost Averaging (DCA)")
    
    lot_balance_col = 'Lot Balance'
    
    # Hitung nilai investasi awal
    df['Total Investment'] = df[lot_balance_col] * df['Avg Price']
    total_investment = df['Total Investment'].sum()
    total_current_value = df['Current Value'].sum()
    total_profit = total_current_value - total_investment
    total_profit_percent = (total_current_value / total_investment - 1) * 100
    
    # Tampilkan metrik utama dalam kolom responsif
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Investasi", f"Rp {total_investment:,.0f}", "Nilai Awal")
    col2.metric("Nilai Saat Ini", f"Rp {total_current_value:,.0f}", 
                f"{total_profit_percent:+.2f}%")
    col3.metric("Profit/Loss", f"Rp {total_profit:,.0f}", 
                f"{total_profit_percent:+.2f}%")
    
    # Grafik alokasi portfolio
    st.subheader("Alokasi Portfolio")
    fig = px.pie(df, names='Ticker', values='Current Value',
                 hover_data=['Profit/Loss %'],
                 title='Komposisi Portfolio Berdasarkan Nilai Saat Ini')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Grafik profit/loss
    st.subheader("Profit/Loss per Saham")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Ticker'],
        y=df['Profit/Loss'],
        text=df['Profit/Loss'].apply(lambda x: f"Rp {x:,.0f}"),
        textposition='auto',
        marker_color=np.where(df['Profit/Loss'] >= 0, 'green', 'red')
    ))
    
    fig.update_layout(
        title='Profit/Loss per Saham',
        yaxis_title='Jumlah (Rp)',
        xaxis_title='Saham'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan tabel detail
    st.subheader("Detail Portfolio")
    df_display = df[['Ticker', lot_balance_col, 'Avg Price', 'Current Price', 
                    'Total Investment', 'Current Value', 'Profit/Loss', 'Profit/Loss %']]
    
    # Rename kolom untuk tampilan yang lebih baik
    df_display = df_display.rename(columns={
        lot_balance_col: 'Jumlah Lembar',
        'Avg Price': 'Harga Rata-rata',
        'Current Price': 'Harga Saat Ini',
        'Total Investment': 'Total Investasi',
        'Current Value': 'Nilai Saat Ini',
        'Profit/Loss': 'Keuntungan/Kerugian',
        'Profit/Loss %': 'Keuntungan/Kerugian %'
    })
    
    st.dataframe(df_display.style.format({
        'Harga Rata-rata': 'Rp {:,.0f}',
        'Harga Saat Ini': 'Rp {:,.0f}',
        'Total Investasi': 'Rp {:,.0f}',
        'Nilai Saat Ini': 'Rp {:,.0f}',
        'Keuntungan/Kerugian': 'Rp {:,.0f}',
        'Keuntungan/Kerugian %': '{:+.2f}%'
    }), use_container_width=True)
    
    return df

# Fungsi untuk menghitung skor valuasi saham
def calculate_valuation_score(ticker, api_key):
    try:
        fmp_data = get_fmp_data(ticker, api_key)
        if not fmp_data:
            return 0
        
        ratios = fmp_data['ratios']
        quote = fmp_data['quote']
        profile = fmp_data['profile']
        
        # Dapatkan rasio yang diperlukan
        per = ratios.get('priceEarningsRatio', 0)
        pbv = ratios.get('priceToBookRatio', 0)
        roe = ratios.get('returnOnEquity', 0) * 100
        npm = ratios.get('netProfitMargin', 0) * 100
        dividend_yield = ratios.get('dividendYield', 0) * 100
        
        # Hitung skor valuasi
        score = 0
        
        # PER rendah lebih baik
        if per > 0 and per < 15:
            score += 3
        elif per < 20:
            score += 2
        elif per < 25:
            score += 1
            
        # PBV rendah lebih baik
        if pbv > 0 and pbv < 1:
            score += 3
        elif pbv < 1.5:
            score += 2
        elif pbv < 2:
            score += 1
            
        # ROE tinggi lebih baik
        if roe > 20:
            score += 3
        elif roe > 15:
            score += 2
        elif roe > 10:
            score += 1
            
        # NPM tinggi lebih baik
        if npm > 20:
            score += 3
        elif npm > 15:
            score += 2
        elif npm > 10:
            score += 1
            
        # Dividend yield tinggi lebih baik
        if dividend_yield > 5:
            score += 3
        elif dividend_yield > 3:
            score += 2
        elif dividend_yield > 1:
            score += 1
            
        return score
    
    except Exception as e:
        st.error(f"Error calculating valuation score: {str(e)}")
        return 0

# Fungsi simulasi pembelian saham berbasis valuasi
def investment_simulation(portfolio_df, api_key):
    st.subheader("💰 Rekomendasi Pembelian Saham Berbasis Valuasi")
    
    # Perbarui data dengan harga real-time
    portfolio_df = update_portfolio_data(portfolio_df)
    
    # Input modal
    investment_amount = st.number_input(
        "Modal Investasi (Rp)", 
        min_value=100000, 
        step=100000, 
        value=500000,
        format="%d"
    )
    
    # Hitung skor valuasi untuk setiap saham
    valuation_scores = []
    for ticker in portfolio_df['Ticker']:
        clean_ticker = ticker.replace('.JK', '')
        score = calculate_valuation_score(clean_ticker, api_key) if api_key else 0
        valuation_scores.append(score)
    
    portfolio_df['Valuation Score'] = valuation_scores
    
    # Urutkan berdasarkan skor valuasi tertinggi
    portfolio_df = portfolio_df.sort_values(by='Valuation Score', ascending=False)
    
    # Hitung bobot alokasi berdasarkan skor
    total_score = portfolio_df['Valuation Score'].sum()
    if total_score > 0:
        portfolio_df['Allocation Weight'] = portfolio_df['Valuation Score'] / total_score
    else:
        # Jika semua skor 0, alokasikan sama rata
        portfolio_df['Allocation Weight'] = 1 / len(portfolio_df)
    
    # Alokasikan dana berdasarkan bobot
    portfolio_df['Allocation Amount'] = portfolio_df['Allocation Weight'] * investment_amount
    portfolio_df['Additional Shares'] = (portfolio_df['Allocation Amount'] / portfolio_df['Current Price']).astype(int)
    portfolio_df['Additional Investment'] = portfolio_df['Additional Shares'] * portfolio_df['Current Price']
    
    # Hitung total yang benar-benar dialokasikan (mungkin ada sisa karena pembulatan)
    actual_investment = portfolio_df['Additional Investment'].sum()
    
    # Hitung sisa dana
    remaining_capital = investment_amount - actual_investment
    
    # Jika ada sisa dana, alokasikan ke saham dengan skor tertinggi
    if remaining_capital > 0:
        for idx, row in portfolio_df.iterrows():
            if remaining_capital <= 0:
                break
            current_price = row['Current Price']
            if current_price <= remaining_capital:
                additional_shares = remaining_capital // current_price
                if additional_shares > 0:
                    portfolio_df.at[idx, 'Additional Shares'] += additional_shares
                    additional_investment = additional_shares * current_price
                    portfolio_df.at[idx, 'Additional Investment'] += additional_investment
                    remaining_capital -= additional_investment
    
    # Hitung nilai baru
    portfolio_df['New Shares'] = portfolio_df['Lot Balance'] + portfolio_df['Additional Shares']
    portfolio_df['New Value'] = portfolio_df['New Shares'] * portfolio_df['Current Price']
    
    # Hitung total setelah simulasi
    total_new_investment = portfolio_df['Additional Investment'].sum()
    total_new_value = portfolio_df['New Value'].sum()
    total_portfolio_value = portfolio_df['Current Value'].sum()
    
    # Tampilkan hasil simulasi
    st.write(f"### Rekomendasi Pembelian untuk Modal Rp {investment_amount:,.0f}")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Investasi Tambahan", f"Rp {total_new_investment:,.0f}")
    col2.metric("Total Nilai Portfolio Baru", f"Rp {total_new_value:,.0f}", 
                f"{((total_new_value - total_portfolio_value)/total_portfolio_value*100):+.2f}%")
    
    # Tampilkan rekomendasi pembelian
    st.subheader("Rekomendasi Pembelian Saham")
    
    # Urutkan berdasarkan jumlah pembelian terbanyak
    buy_recommendations = portfolio_df[portfolio_df['Additional Shares'] > 0].copy()
    buy_recommendations = buy_recommendations.sort_values(by='Additional Investment', ascending=False)
    
    if not buy_recommendations.empty:
        # Hitung rangking
        buy_recommendations['Ranking'] = range(1, len(buy_recommendations) + 1)
        
        # Tampilkan tabel rekomendasi
        rec_df = buy_recommendations[[
            'Ranking', 'Ticker', 'Valuation Score', 'Current Price', 
            'Additional Shares', 'Additional Investment'
        ]]
        
        # Rename kolom
        rec_df = rec_df.rename(columns={
            'Valuation Score': 'Skor Valuasi',
            'Current Price': 'Harga Saat Ini',
            'Additional Shares': 'Jumlah Pembelian',
            'Additional Investment': 'Total Pembelian'
        })
        
        # Format kolom
        rec_display = rec_df.style.format({
            'Harga Saat Ini': 'Rp {:,.0f}',
            'Total Pembelian': 'Rp {:,.0f}'
        }).background_gradient(subset=['Skor Valuasi'], cmap='YlGn')
        
        st.dataframe(rec_display, use_container_width=True)
        
        # Grafik alokasi pembelian
        st.subheader("Alokasi Pembelian")
        fig = px.pie(buy_recommendations, names='Ticker', values='Additional Investment',
                     title='Distribusi Pembelian Berdasarkan Valuasi')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Grafik perbandingan skor valuasi
        st.subheader("Skor Valuasi Saham")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=buy_recommendations['Ticker'],
            y=buy_recommendations['Valuation Score'],
            text=buy_recommendations['Valuation Score'],
            textposition='auto',
            marker_color='skyblue'
        ))
        fig.update_layout(
            title='Skor Valuasi Saham',
            yaxis_title='Skor',
            xaxis_title='Saham'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada rekomendasi pembelian saham dengan modal yang tersedia")
    
    return portfolio_df

# Fungsi prediksi harga dengan model ARIMA
def stock_prediction(ticker):
    try:
        st.subheader(f"📈 Prediksi Harga Saham: {ticker}")
        
        # Dapatkan data historis
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            st.warning(f"Data tidak ditemukan untuk {ticker}")
            return
        
        # Tampilkan grafik harga historis
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Harga Historis'
        ))
        
        # Tambahkan moving averages
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MA20'],
            line=dict(color='orange', width=1.5),
            name='MA 20 Hari'
        ))
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MA50'],
            line=dict(color='blue', width=1.5),
            name='MA 50 Hari'
        ))
        
        fig.update_layout(
            title=f'Perjalanan Harga {ticker}',
            yaxis_title='Harga (Rp)',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediksi sederhana menggunakan moving average
        last_price = hist['Close'].iloc[-1]
        ma20 = hist['MA20'].iloc[-1]
        ma50 = hist['MA50'].iloc[-1]
        
        # Logika prediksi sederhana
        if ma20 > ma50 and last_price > ma20:
            trend = "Naik"
            prediction = last_price * 1.05  # +5%
        elif ma20 < ma50 and last_price < ma20:
            trend = "Turun"
            prediction = last_price * 0.95  # -5%
        else:
            trend = "Netral"
            prediction = last_price * 1.01  # +1%
        
        # Tampilkan metrik prediksi
        col1, col2, col3 = st.columns(3)
        col1.metric("Harga Terakhir", f"Rp {last_price:,.0f}")
        col2.metric("Prediksi 1 Bulan", f"Rp {prediction:,.0f}", 
                   f"{(prediction/last_price-1)*100:+.2f}%")
        col3.metric("Trend", trend)
        
        # Analisis teknis tambahan
        st.subheader("Analisis Teknis")
        rsi = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=rsi, name='RSI', line=dict(color='purple')))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.update_layout(title='Relative Strength Index (RSI)', yaxis_title='RSI')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=macd, name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hist.index, y=signal, name='Signal', line=dict(color='orange')))
        fig.update_layout(title='Moving Average Convergence Divergence (MACD)', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Fungsi untuk menghitung RSI
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fungsi untuk menghitung MACD
def calculate_macd(prices, slow=26, fast=12, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

# Fungsi valuasi saham dengan data FMP
def stock_valuation(ticker, api_key):
    try:
        st.subheader(f"💰 Valuasi Saham: {ticker}")
        
        # Dapatkan data dari FMP
        fmp_data = get_fmp_data(ticker, api_key)
        if not fmp_data:
            st.warning("Tidak dapat melanjutkan valuasi tanpa data FMP")
            return
        
        # Ekstrak data yang dibutuhkan
        profile = fmp_data['profile']
        ratios = fmp_data['ratios']
        cashflow = fmp_data['cashflow']
        quote = fmp_data['quote']
        growth = fmp_data['growth']
        
        # Tampilkan metrik utama
        col1, col2, col3 = st.columns(3)
        current_price = quote.get('price', 0)
        previous_close = quote.get('previousClose', current_price)
        change = quote.get('change', 0)
        change_percent = quote.get('changesPercentage', 0)
        
        col1.metric("Harga Saat Ini", f"Rp {current_price:,.0f}", 
                   f"{change_percent:+.2f}%")
        
        # PER (Price to Earnings Ratio)
        per = ratios.get('priceEarningsRatio', 0)
        industry_per = profile.get('peRatio', per * 1.1)  # Contoh perbandingan industri
        col2.metric("PER (Price/Earnings)", f"{per:.2f}", 
                   f"Industri: {industry_per:.2f}", delta_color="off")
        
        # PBV (Price to Book Value)
        pbv = ratios.get('priceToBookRatio', 0)
        industry_pbv = pbv * 1.15  # Contoh perbandingan industri
        col3.metric("PBV (Price/Book)", f"{pbv:.2f}", 
                   f"Industri: {industry_pbv:.2f}", delta_color="off")
        
        # Rasio keuangan lainnya
        st.subheader("Rasio Keuangan")
        ratios_data = {
            'ROE': ratios.get('returnOnEquity', 0) * 100,
            'DER': ratios.get('debtEquityRatio', 0),
            'NPM': ratios.get('netProfitMargin', 0) * 100,
            'EPS': ratios.get('earningsPerShare', 0),
            'Dividend Yield': ratios.get('dividendYield', 0) * 100,
            'Revenue Growth': growth.get('growthRevenue', 0) * 100
        }
        
        # Tampilkan dalam bentuk metrik
        cols = st.columns(len(ratios_data))
        for i, (name, value) in enumerate(ratios_data.items()):
            cols[i].metric(name, f"{value:.2f}{'%' if name != 'EPS' else ''}")
        
        # Grafik perbandingan valuasi
        st.subheader("Perbandingan Valuasi")
        valuation_data = {
            'Metric': ['PER', 'PBV', 'ROE', 'DER', 'NPM', 'Dividend Yield'],
            'Nilai': [per, pbv, ratios_data['ROE'], ratios_data['DER'], 
                     ratios_data['NPM'], ratios_data['Dividend Yield']],
            'Rata-rata Industri': [industry_per, industry_pbv, 
                                  ratios_data['ROE'] * 0.9, ratios_data['DER'] * 1.1, 
                                  ratios_data['NPM'] * 0.95, ratios_data['Dividend Yield'] * 1.2]
        }
        
        df_valuation = pd.DataFrame(valuation_data)
        fig = px.bar(df_valuation, x='Metric', y=['Nilai', 'Rata-rata Industri'], 
                     barmode='group', title='Perbandingan Valuasi dengan Rata-rata Industri')
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis DCF menggunakan data FMP
        st.subheader("Discounted Cash Flow (DCF)")
        
        # Dapatkan free cash flow
        free_cash_flow = cashflow.get('freeCashFlow', 0)
        
        # Hitung pertumbuhan historis
        growth_rate = ratios_data['Revenue Growth'] / 100
        if growth_rate > 0.15:
            growth_rate = 0.15  # Batasi pertumbuhan tinggi
        elif growth_rate < 0.03:
            growth_rate = 0.03  # Minimum growth
        
        # Asumsi
        st.write(f"""
        **Asumsi:**
        - Pertumbuhan 5 tahun: {growth_rate*100:.1f}%
        - Pertumbuhan terminal: 3.0%
        - Tingkat diskonto: 10.0%
        - Free Cash Flow terakhir: Rp {free_cash_flow:,.0f}
        """)
        
        # Hitung valuasi DCF
        terminal_growth = 0.03
        discount_rate = 0.10
        
        # Proyeksi FCF untuk 5 tahun
        future_fcf = [free_cash_flow * (1 + growth_rate) ** i for i in range(1, 6)]
        
        # Terminal value
        terminal_value = future_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        
        # Discount factor
        discount_factors = [1 / (1 + discount_rate) ** i for i in range(1, 6)]
        
        # Nilai sekarang arus kas
        pv_cash_flows = [fcf * df for fcf, df in zip(future_fcf, discount_factors)]
        pv_terminal = terminal_value * discount_factors[-1]
        
        # Total nilai perusahaan
        enterprise_value = sum(pv_cash_flows) + pv_terminal
        
        # Nilai ekuitas (dikurangi hutang, tambah kas)
        debt = profile.get('totalDebt', 0)
        cash = profile.get('cash', 0)
        equity_value = enterprise_value - debt + cash
        
        # Nilai per saham
        shares = profile.get('outstandingShares', 1)
        fair_value = equity_value / shares
        
        # Tampilkan hasil
        col1, col2 = st.columns(2)
        col1.metric("Nilai Wajar (DCF)", f"Rp {fair_value:,.0f}")
        col2.metric("Premium/Diskon", 
                   f"{(current_price/fair_value-1)*100:+.2f}%", 
                   "vs Harga Saat Ini")
        
        # Grafik DCF
        dcf_df = pd.DataFrame({
            'Tahun': ['Tahun 1', 'Tahun 2', 'Tahun 3', 'Tahun 4', 'Tahun 5', 'Terminal'],
            'Nilai': pv_cash_flows + [pv_terminal]
        })
        
        fig = px.pie(dcf_df, names='Tahun', values='Nilai', 
                     title='Komposisi Nilai DCF')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in valuation: {str(e)}")
        st.error(f"Detail error: {e}")

# Fungsi tracking modal
def capital_tracking():
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []
        
    st.subheader("💵 Tracking Modal")
    
    with st.expander("Tambah Transaksi Baru"):
        with st.form("transaction_form"):
            date = st.date_input("Tanggal Transaksi", datetime.today())
            ticker = st.text_input("Kode Saham", "BBCA.JK")
            action = st.selectbox("Aksi", ["Beli", "Jual"])
            shares = st.number_input("Jumlah Lembar", min_value=1, value=100)
            price = st.number_input("Harga per Lembar (Rp)", min_value=1, value=10000)
            submit = st.form_submit_button("Tambahkan Transaksi")
            
            if submit:
                transaction = {
                    'Date': date,
                    'Ticker': ticker,
                    'Action': action,
                    'Shares': shares,
                    'Price': price,
                    'Amount': shares * price * (-1 if action == "Jual" else 1)
                }
                st.session_state.transactions.append(transaction)
                st.success("Transaksi ditambahkan!")
    
    if st.session_state.transactions:
        df_transactions = pd.DataFrame(st.session_state.transactions)
        
        # Hitung saldo
        df_transactions['Cumulative'] = df_transactions['Amount'].cumsum()
        
        # Tampilkan tabel transaksi
        st.dataframe(df_transactions.style.format({
            'Price': 'Rp {:,.0f}',
            'Amount': 'Rp {:,.0f}',
            'Cumulative': 'Rp {:,.0f}'
        }), use_container_width=True)
        
        # Grafik cashflow
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_transactions['Date'],
            y=df_transactions['Cumulative'],
            mode='lines+markers',
            name='Saldo Akumulatif'
        ))
        fig.update_layout(
            title='Riwayat Saldo Investasi',
            yaxis_title='Saldo (Rp)',
            xaxis_title='Tanggal'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Ringkasan modal
        total_investment = df_transactions[df_transactions['Action'] == 'Beli']['Amount'].sum()
        total_sales = abs(df_transactions[df_transactions['Action'] == 'Jual']['Amount'].sum())
        net_cashflow = df_transactions['Amount'].sum()
        current_balance = net_cashflow
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pembelian", f"Rp {total_investment:,.0f}")
        col2.metric("Total Penjualan", f"Rp {total_sales:,.0f}")
        col3.metric("Saldo Saat Ini", f"Rp {current_balance:,.0f}")

# Fungsi untuk mendapatkan NewsAPI key
def get_news_api_key():
    if 'news_api_key' not in st.session_state:
        with st.sidebar:
            st.subheader("NewsAPI Configuration")
            api_key = st.text_input("Masukkan NewsAPI Key", type="password")
            if st.button("Simpan NewsAPI Key"):
                st.session_state.news_api_key = api_key
                st.success("NewsAPI Key disimpan!")
                st.rerun()
        return None
    return st.session_state.news_api_key

# Fungsi untuk mendapatkan berita dari NewsAPI
def get_news_from_newsapi(query, api_key, language='en', page_size=10):
    try:
        # Periksa ketersediaan modul
        if NewsApiClient is None:
            st.warning("NewsAPI client tidak tersedia. Gunakan sumber berita lain.")
            return []
            
        newsapi = NewsApiClient(api_key=api_key)
        news = newsapi.get_everything(q=query,
                                     language=language,
                                     sort_by='relevancy',
                                     page_size=page_size)
        articles = []
        for article in news['articles']:
            articles.append({
                'title': article['title'],
                'description': article['description'],
                'url': article['url'],
                'source': article['source']['name'],
                'published_at': article['published_at'],
                'content': article['content']
            })
        return articles
    except Exception as e:
        st.error(f"Error fetching news from NewsAPI: {str(e)}")
        return []

# Fungsi untuk mendapatkan berita dari Yahoo Finance
def get_news_from_yahoo(ticker):
    try:
        news_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(news_url)
        articles = []
        for entry in feed.entries[:10]:
            articles.append({
                'title': entry.title,
                'description': entry.summary if 'summary' in entry else '',
                'url': entry.link,
                'source': 'Yahoo Finance',
                'published_at': entry.published if 'published' in entry else '',
                'content': entry.summary if 'summary' in entry else ''
            })
        return articles
    except Exception as e:
        st.error(f"Error fetching news from Yahoo Finance: {str(e)}")
        return []

# Fungsi analisis sentimen
def analyze_sentiment(text):
    try:
        # Analisis dengan VADER
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(text)
        
        # Analisis dengan TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Gabungkan hasil
        sentiment = {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': polarity,
            'textblob_subjectivity': subjectivity,
            'combined_score': (vader_scores['compound'] + polarity) / 2
        }
        return sentiment
    except:
        return {
            'vader_compound': 0,
            'vader_positive': 0,
            'vader_negative': 0,
            'vader_neutral': 0,
            'textblob_polarity': 0,
            'textblob_subjectivity': 0,
            'combined_score': 0
        }

# Fungsi untuk menampilkan heatmap sentimen sektoral
def display_sector_sentiment_heatmap(sector_data):
    if not sector_data:
        st.warning("Tidak ada data sentimen sektoral yang tersedia")
        return
    
    # Siapkan data untuk heatmap
    sectors = list(sector_data.keys())
    sentiment_scores = [sector_data[sector]['average_sentiment'] for sector in sectors]
    news_counts = [sector_data[sector]['count'] for sector in sectors]
    
    # Buat heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[sentiment_scores],
        x=sectors,
        y=['Sentimen'],
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Sentimen'),
        hoverongaps=False,
        text=[f"Sektor: {sector}<br>Sentimen: {score:.2f}<br>Berita: {count}" 
              for sector, score, count in zip(sectors, sentiment_scores, news_counts)],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Heatmap Sentimen Pasar per Sektor',
        xaxis_title='Sektor',
        yaxis_title='',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk menampilkan news feed - DIUBAH UNTUK RESPONSIF
def display_news_feed():
    st.subheader("📰 Market News & Sentiment Analysis")
    
    # Dapatkan API keys
    news_api_key = get_news_api_key()
    fmp_api_key = get_fmp_api_key()
    
    # Pilihan sumber berita
    col1, col2 = st.columns(2)
    with col1:
        news_source = st.selectbox("Sumber Berita", ["Yahoo Finance", "NewsAPI"])
    with col2:
        if news_source == "NewsAPI":
            if not news_api_key:
                st.warning("Masukkan NewsAPI Key di sidebar")
            elif NewsApiClient is None:
                st.warning("NewsAPI client tidak tersedia")
    
    # Analisis sentimen tingkat
    analysis_level = st.radio("Tingkat Analisis", ["Market", "Sektor", "Saham Tertentu"], horizontal=True)
    
    # Kumpulkan berita berdasarkan tingkat analisis
    articles = []
    sector_sentiment = {}
    
    if analysis_level == "Market":
        query = "stocks OR market OR economy"
        if news_source == "NewsAPI" and news_api_key and NewsApiClient:
            articles = get_news_from_newsapi(query, news_api_key)
        else:
            articles = get_news_from_yahoo('^GSPC')  # S&P 500 sebagai proxy market
    
    elif analysis_level == "Sektor":
        # Dapatkan daftar sektor dari FMP
        sectors = []
        if fmp_api_key:
            try:
                sectors_url = f"https://financialmodelingprep.com/api/v3/sector-performance?apikey={fmp_api_key}"
                response = requests.get(sectors_url)
                sectors_data = response.json()
                sectors = [sector['sector'] for sector in sectors_data]
            except:
                sectors = ["Technology", "Financial Services", "Healthcare", "Energy", 
                          "Consumer Cyclical", "Industrials", "Communication Services"]
        
        selected_sector = st.selectbox("Pilih Sektor", sectors)
        
        if news_source == "NewsAPI" and news_api_key and NewsApiClient:
            articles = get_news_from_newsapi(f"{selected_sector} sector", news_api_key)
        else:
            # Untuk Yahoo, coba cari berdasarkan sektor
            articles = get_news_from_yahoo(selected_sector.split()[0])
    
    elif analysis_level == "Saham Tertentu":
        ticker = st.text_input("Masukkan Kode Saham (contoh: AAPL)", "AAPL")
        if news_source == "NewsAPI" and news_api_key and NewsApiClient:
            articles = get_news_from_newsapi(ticker, news_api_key)
        else:
            articles = get_news_from_yahoo(ticker)
    
    # Proses dan tampilkan berita
    if articles:
        st.subheader(f"Berita Terbaru ({len(articles)} ditemukan)")
        
        # Analisis sentimen untuk setiap artikel
        sentiment_scores = []
        for article in articles:
            text = f"{article['title']}. {article['description']}"
            sentiment = analyze_sentiment(text)
            article['sentiment'] = sentiment
            sentiment_scores.append(sentiment['combined_score'])
        
        # Hitung rata-rata sentimen
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Tampilkan metrik sentimen
        sentiment_color = "green" if avg_sentiment > 0.1 else "red" if avg_sentiment < -0.1 else "gray"
        st.metric("Rata-rata Sentimen Pasar", 
                 f"{avg_sentiment:.2f}", 
                 "Positif" if avg_sentiment > 0.1 else "Negatif" if avg_sentiment < -0.1 else "Netral",
                 delta_color="off")
        
        # Heatmap sentimen sektoral
        if analysis_level == "Market" and fmp_api_key:
            try:
                # Dapatkan performa sektor
                sectors_url = f"https://financialmodelingprep.com/api/v3/sector-performance?apikey={fmp_api_key}"
                response = requests.get(sectors_url)
                sectors_data = response.json()
                
                # Analisis sentimen per sektor
                sector_sentiment = {}
                for sector in sectors_data:
                    sector_name = sector['sector']
                    if news_source == "NewsAPI" and news_api_key and NewsApiClient:
                        sector_articles = get_news_from_newsapi(sector_name, news_api_key, page_size=5)
                    else:
                        sector_articles = get_news_from_yahoo(sector_name.split()[0])
                    
                    sector_scores = []
                    for article in sector_articles:
                        text = f"{article['title']}. {article['description']}"
                        sentiment = analyze_sentiment(text)
                        sector_scores.append(sentiment['combined_score'])
                    
                    avg_sentiment = np.mean(sector_scores) if sector_scores else 0
                    sector_sentiment[sector_name] = {
                        'average_sentiment': avg_sentiment,
                        'count': len(sector_articles)
                    }
                
                display_sector_sentiment_heatmap(sector_sentiment)
            except Exception as e:
                st.error(f"Error generating sector sentiment: {str(e)}")
        
        # Tampilkan artikel dengan analisis sentimen
        st.subheader("Berita Terbaru dengan Analisis Sentimen")
        for article in articles:
            sentiment = article['sentiment']
            sentiment_score = sentiment['combined_score']
            
            # Tentukan warna berdasarkan sentimen
            if sentiment_score > 0.2:
                border_color = "green"
            elif sentiment_score < -0.2:
                border_color = "red"
            else:
                border_color = "gray"
            
            # Tampilkan artikel dalam container
            with st.container():
                st.markdown(f"""
                <div style="
                    border-left: 5px solid {border_color};
                    padding: 10px;
                    margin-bottom: 10px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                ">
                    <h4><a href="{article['url']}" target="_blank">{article['title']}</a></h4>
                    <p>{article['description']}</p>
                    <div style="font-size: 0.8em; color: #666;">
                        <b>Sumber:</b> {article['source']} | 
                        <b>Tanggal:</b> {article['published_at'][:10] if article['published_at'] else 'N/A'} | 
                        <b>Sentimen:</b> {sentiment_score:.2f} 
                        <span style="color: {'green' if sentiment_score > 0 else 'red' if sentiment_score < 0 else 'gray'}">
                        ({'Positif' if sentiment_score > 0.2 else 'Sangat Positif' if sentiment_score > 0.5 else 
                          'Negatif' if sentiment_score < -0.2 else 'Sangat Negatif' if sentiment_score < -0.5 else 
                          'Netral'})
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Tidak ada berita yang ditemukan. Coba sumber atau kueri yang berbeda.")

# Fungsi untuk mendapatkan rekomendasi saham undervalued
def get_undervalued_recommendations(api_key):
    st.subheader("🔍 Saham Undervalued Minggu Ini")
    
    if not api_key:
        st.warning("Silakan masukkan API Key FMP di sidebar untuk mengakses fitur ini")
        return []
    
    try:
        # Dapatkan daftar saham Indonesia
        indonesian_stocks_url = f"https://financialmodelingprep.com/api/v3/stock-screener?exchange=IDX&apikey={api_key}"
        response = requests.get(indonesian_stocks_url)
        stocks_data = response.json()
        
        if not stocks_data:
            st.warning("Tidak dapat menemukan data saham Indonesia")
            return []
        
        # Filter saham dengan market cap > 1T
        filtered_stocks = [
            stock for stock in stocks_data 
            if stock.get('marketCap', 0) > 1000000000000
        ]
        
        # Batasi jumlah saham untuk efisiensi
        selected_stocks = random.sample(filtered_stocks, min(20, len(filtered_stocks)))
        
        undervalued_stocks = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, stock in enumerate(selected_stocks):
            ticker = stock['symbol']
            status_text.text(f"Menganalisis {ticker} ({i+1}/{len(selected_stocks)})...")
            
            # Dapatkan data valuasi
            fmp_data = get_fmp_data(ticker, api_key)
            if not fmp_data:
                continue
                
            ratios = fmp_data.get('ratios', {})
            quote = fmp_data.get('quote', {})
            
            # Kriteria undervalued
            per = ratios.get('priceEarningsRatio', 100)
            pbv = ratios.get('priceToBookRatio', 10)
            dividend_yield = ratios.get('dividendYield', 0) * 100
            roe = ratios.get('returnOnEquity', 0) * 100
            
            # Hitung skor undervalued
            score = 0
            
            # PER rendah lebih baik
            if per < 10:
                score += 3
            elif per < 15:
                score += 2
            elif per < 20:
                score += 1
                
            # PBV rendah lebih baik
            if pbv < 1:
                score += 3
            elif pbv < 1.5:
                score += 2
            elif pbv < 2:
                score += 1
                
            # Dividend yield tinggi lebih baik
            if dividend_yield > 5:
                score += 3
            elif dividend_yield > 3:
                score += 2
            elif dividend_yield > 1:
                score += 1
                
            # ROE tinggi lebih baik
            if roe > 20:
                score += 3
            elif roe > 15:
                score += 2
            elif roe > 10:
                score += 1
                
            # Tambahkan jika memenuhi kriteria
            if score >= 8:  # Threshold untuk undervalued
                undervalued_stocks.append({
                    'Ticker': ticker,
                    'Nama': stock.get('companyName', ticker),
                    'PER': per,
                    'PBV': pbv,
                    'Dividend Yield': dividend_yield,
                    'ROE': roe,
                    'Skor': score,
                    'Harga': quote.get('price', 0)
                })
            
            progress_bar.progress((i+1)/len(selected_stocks))
        
        status_text.text("Analisis selesai!")
        
        if undervalued_stocks:
            # Urutkan berdasarkan skor tertinggi
            undervalued_stocks.sort(key=lambda x: x['Skor'], reverse=True)
            
            # Tampilkan hasil
            st.success(f"Ditemukan {len(undervalued_stocks)} saham undervalued!")
            
            # Tampilkan dalam tabel
            df = pd.DataFrame(undervalued_stocks)
            st.dataframe(df.style.format({
                'PER': '{:.2f}',
                'PBV': '{:.2f}',
                'Dividend Yield': '{:.2f}%',
                'ROE': '{:.2f}%',
                'Harga': 'Rp {:,.0f}'
            }).background_gradient(subset=['Skor'], cmap='YlGn'), use_container_width=True)
            
            # Tampilkan grafik perbandingan
            st.subheader("Perbandingan Saham Undervalued")
            fig = px.bar(df, x='Nama', y='Skor', color='Skor',
                         title='Skor Undervalued Saham',
                         labels={'Nama': 'Saham', 'Skor': 'Skor Undervalued'})
            st.plotly_chart(fig, use_container_width=True)
            
            return undervalued_stocks
        else:
            st.warning("Tidak ditemukan saham yang memenuhi kriteria undervalued minggu ini")
            return []
            
    except Exception as e:
        st.error(f"Error dalam mendapatkan rekomendasi saham undervalued: {str(e)}")
        return []

# Fungsi untuk menentukan profil risiko pengguna - DIUBAH UNTUK RESPONSIF
def get_risk_profile():
    st.subheader("📝 Profil Risiko Investor")
    
    if 'risk_profile' not in st.session_state:
        with st.form("risk_profile_form"):
            st.write("Silakan jawab pertanyaan berikut untuk menentukan profil risiko:")
            
            # Gunakan radio dengan layout vertikal untuk mobile
            q1 = st.radio(
                "1. Apa tujuan utama investasi Anda?",
                options=[
                    ("Pelestarian modal (risiko rendah)", 1),
                    ("Pertumbuhan modal moderat", 2),
                    ("Pertumbuhan modal agresif", 3),
                    ("Pendapatan spekulatif tinggi", 4)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            q2 = st.radio(
                "2. Berapa lama horizon investasi Anda?",
                options=[
                    ("Kurang dari 1 tahun", 1),
                    ("1-3 tahun", 2),
                    ("3-5 tahun", 3),
                    ("Lebih dari 5 tahun", 4)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            q3 = st.radio(
                "3. Bagaimana reaksi Anda terhadap penurunan 20% portofolio dalam 1 bulan?",
                options=[
                    ("Jual semua investasi", 1),
                    ("Jual sebagian", 2),
                    ("Tahan dan pantau", 3),
                    ("Beli lebih banyak", 4)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            q4 = st.radio(
                "4. Pengalaman investasi Anda?",
                options=[
                    ("Pemula (baru mulai)", 1),
                    ("Sedang (1-3 tahun)", 2),
                    ("Berpengalaman (3-5 tahun)", 3),
                    ("Sangat berpengalaman (>5 tahun)", 4)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            submitted = st.form_submit_button("Tentukan Profil Risiko")
            
            if submitted:
                total_score = q1 + q2 + q3 + q4
                
                if total_score <= 6:
                    profile = "Konservatif"
                elif total_score <= 10:
                    profile = "Moderat"
                elif total_score <= 14:
                    profile = "Agresif"
                else:
                    profile = "Sangat Agresif"
                
                st.session_state.risk_profile = profile
                st.success(f"Profil risiko Anda: {profile}")
                st.rerun()
    
    if 'risk_profile' in st.session_state:
        st.info(f"Profil risiko saat ini: **{st.session_state.risk_profile}**")
        if st.button("Ubah Profil Risiko"):
            del st.session_state.risk_profile
            st.rerun()
    
    return st.session_state.get('risk_profile', None)

# Fungsi untuk memberikan rekomendasi diversifikasi
def get_diversification_recommendation(portfolio_df, risk_profile):
    st.subheader("🌐 Rekomendasi Diversifikasi Portofolio")
    
    if not risk_profile:
        st.warning("Silakan tentukan profil risiko Anda terlebih dahulu")
        return
    
    if portfolio_df.empty:
        st.warning("Silakan upload portofolio Anda terlebih dahulu")
        return
    
    # Standar alokasi berdasarkan profil risiko
    allocation_guidelines = {
        "Konservatif": {
            "Saham Blue Chip": 70,
            "Saham Pendapatan": 20,
            "Reksa Dana Pendapatan Tetap": 10,
            "Saham Growth": 0,
            "Saham Spekulatif": 0
        },
        "Moderat": {
            "Saham Blue Chip": 50,
            "Saham Pendapatan": 20,
            "Reksa Dana Pendapatan Tetap": 10,
            "Saham Growth": 15,
            "Saham Spekulatif": 5
        },
        "Agresif": {
            "Saham Blue Chip": 30,
            "Saham Pendapatan": 10,
            "Reksa Dana Pendapatan Tetap": 5,
            "Saham Growth": 40,
            "Saham Spekulatif": 15
        },
        "Sangat Agresif": {
            "Saham Blue Chip": 20,
            "Saham Pendapatan": 5,
            "Reksa Dana Pendapatan Tetap": 0,
            "Saham Growth": 50,
            "Saham Spekulatif": 25
        }
    }
    
    # Klasifikasi saham (sederhana)
    blue_chips = ["BBCA", "BBRI", "BBNI", "BMRI", "TLKM", "EXCL", "ASII"]
    income_stocks = ["UNVR", "ICBP", "MYOR", "INDF", "SMGR"]
    growth_stocks = ["GOTO", "ARTO", "BRIS", "ACES", "EMTK"]
    
    # Kategorikan saham di portofolio
    portfolio_df['Kategori'] = "Lainnya"
    
    for idx, row in portfolio_df.iterrows():
        ticker = row['Ticker'].replace('.JK', '')
        
        if ticker in blue_chips:
            portfolio_df.at[idx, 'Kategori'] = "Saham Blue Chip"
        elif ticker in income_stocks:
            portfolio_df.at[idx, 'Kategori'] = "Saham Pendapatan"
        elif ticker in growth_stocks:
            portfolio_df.at[idx, 'Kategori'] = "Saham Growth"
        else:
            portfolio_df.at[idx, 'Kategori'] = "Saham Spekulatif"
    
    # Hitung alokasi saat ini
    total_value = portfolio_df['Current Value'].sum()
    current_allocation = portfolio_df.groupby('Kategori')['Current Value'].sum() / total_value * 100
    
    # Dapatkan alokasi target
    target_allocation = allocation_guidelines[risk_profile]
    
    # Tampilkan perbandingan
    st.write(f"### Alokasi Portofolio Saat Ini vs Rekomendasi ({risk_profile})")
    
    # Siapkan data untuk visualisasi
    allocation_data = []
    for category in target_allocation:
        current_pct = current_allocation.get(category, 0)
        target_pct = target_allocation[category]
        allocation_data.append({
            'Kategori': category,
            'Saat Ini': current_pct,
            'Target': target_pct
        })
    
    df_allocation = pd.DataFrame(allocation_data)
    
    # Tampilkan grafik
    fig = px.bar(df_allocation, x='Kategori', y=['Saat Ini', 'Target'],
                 barmode='group', title='Alokasi Portofolio Saat Ini vs Target')
    fig.update_layout(yaxis_title='Persentase (%)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Rekomendasi penyesuaian
    st.subheader("Rekomendasi Penyesuaian")
    
    recommendations = []
    for category in target_allocation:
        current_pct = current_allocation.get(category, 0)
        target_pct = target_allocation[category]
        difference = target_pct - current_pct
        
        if difference > 5:  # Hanya rekomendasikan jika perbedaan signifikan
            action = "Tambah alokasi"
        elif difference < -5:
            action = "Kurangi alokasi"
        else:
            continue
        
        recommendations.append({
            'Kategori': category,
            'Saat Ini': f"{current_pct:.1f}%",
            'Target': f"{target_pct:.1f}%",
            'Aksi': action,
            'Jumlah': f"{abs(difference):.1f}%"
        })
    
    if recommendations:
        df_rec = pd.DataFrame(recommendations)
        st.dataframe(df_rec, use_container_width=True)
    else:
        st.success("Portofolio Anda sudah sesuai dengan alokasi target untuk profil risiko Anda!")
# ==========================================
# FUNGSI KOMPARASI SAHAM BARU
# ==========================================

def stock_comparison(api_key, portfolio_df=pd.DataFrame()):
    st.subheader("📊 Komparasi Saham")
    st.info("Bandingkan saham dari portofolio Anda dengan saham lainnya di pasar Indonesia")
    
    # Dapatkan daftar saham Indonesia
    idx_stocks = []
    if api_key:
        try:
            indonesian_stocks_url = f"https://financialmodelingprep.com/api/v3/stock-screener?exchange=IDX&apikey={api_key}"
            response = requests.get(indonesian_stocks_url)
            stocks_data = response.json()
            idx_stocks = [{"Ticker": stock['symbol'], "Nama": stock['companyName']} for stock in stocks_data]
        except Exception as e:
            st.error(f"Gagal mendapatkan daftar saham Indonesia: {str(e)}")
    
    # Pilihan saham dari portofolio user
    portfolio_options = []
    if not portfolio_df.empty:
        portfolio_options = portfolio_df['Ticker'].unique().tolist()
    
    # UI untuk memilih saham
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Saham Portofolio Anda")
        selected_portfolio = st.multiselect(
            "Pilih saham dari portofolio Anda:",
            options=portfolio_options,
            default=portfolio_options[:min(2, len(portfolio_options))] if portfolio_options else []
        )
    
    with col2:
        st.subheader("Saham Pasar Indonesia")
        # Buat daftar pilihan saham IDX
        idx_options = [f"{stock['Ticker']} ({stock['Nama']})" for stock in idx_stocks]
        selected_idx = st.multiselect(
            "Pilih saham dari pasar Indonesia:",
            options=idx_options,
            default=idx_options[:min(2, len(idx_options))] if idx_options else []
        )
        # Ekstrak kode ticker dari pilihan
        selected_idx_tickers = [option.split(' ')[0] for option in selected_idx]
    
    # Gabungkan semua ticker yang dipilih
    all_tickers = selected_portfolio + selected_idx_tickers
    
    if not all_tickers:
        st.warning("Silakan pilih minimal satu saham untuk dibandingkan")
        return
    
    # Batasi maksimal 5 saham untuk efisiensi
    if len(all_tickers) > 5:
        st.warning("Maksimal 5 saham yang dapat dibandingkan")
        all_tickers = all_tickers[:5]
    
    # Kumpulkan data untuk setiap saham
    comparison_data = []
    
    with st.spinner("Mengumpulkan data saham..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(all_tickers):
            status_text.text(f"Menganalisis {ticker} ({i+1}/{len(all_tickers)})...")
            clean_ticker = ticker.replace('.JK', '')
            
            # Dapatkan data fundamental
            fmp_data = get_fmp_data(clean_ticker, api_key) if api_key else {}
            
            # Dapatkan data sentimen
            sentiment_score = get_stock_sentiment(ticker)
            
            # Ekstrak metrik penting
            profile = fmp_data.get('profile', {}) if fmp_data else {}
            ratios = fmp_data.get('ratios', {}) if fmp_data else {}
            growth = fmp_data.get('growth', {}) if fmp_data else {}
            quote = fmp_data.get('quote', {}) if fmp_data else {}
            
            # Dapatkan harga real-time sebagai fallback
            last_price, _, _, _ = get_realtime_data(ticker)
            
            comparison_data.append({
                "Ticker": ticker,
                "Nama": profile.get('companyName', ticker),
                "Harga": quote.get('price', last_price) if last_price else 0,
                "PER": ratios.get('priceEarningsRatio', 0),
                "PBV": ratios.get('priceToBookRatio', 0),
                "ROE": ratios.get('returnOnEquity', 0) * 100 if ratios.get('returnOnEquity') else 0,
                "Pertumbuhan Pendapatan": growth.get('growthRevenue', 0) * 100 if growth.get('growthRevenue') else 0,
                "Sentimen": sentiment_score,
                "Dividen Yield": ratios.get('dividendYield', 0) * 100 if ratios.get('dividendYield') else 0,
                "Sumber": "Portofolio Anda" if ticker in selected_portfolio else "Pasar Indonesia"
            })
            
            progress_bar.progress((i+1)/len(all_tickers))
    
    if not comparison_data:
        st.error("Tidak ada data yang berhasil dikumpulkan")
        return
    
    # Tampilkan tabel perbandingan
    st.subheader("Perbandingan Metrik Fundamental")
    df = pd.DataFrame(comparison_data)
    
    # Format kolom
    formatted_df = df.copy()
    for col in ["Harga", "PER", "PBV", "ROE", "Pertumbuhan Pendapatan", "Dividen Yield"]:
        if col == "Harga":
            formatted_df[col] = formatted_df[col].apply(lambda x: f"Rp {x:,.0f}" if x and pd.notnull(x) else "-")
        elif col == "Sentimen":
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")
        else:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%" if x and pd.notnull(x) else "-")
    
    # Tampilkan dengan warna berdasarkan sumber
    def color_source(row):
        color = 'lightgreen' if row['Sumber'] == 'Portofolio Anda' else 'lightblue'
        return [f'background-color: {color}'] * len(row)
    
    st.dataframe(
        formatted_df.style.apply(color_source, axis=1),
        use_container_width=True
    )
    
    # Grafik perbandingan
    st.subheader("Grafik Perbandingan")
    
    # Pilih metrik untuk grafik
    metrics = st.multiselect(
        "Pilih metrik untuk ditampilkan:",
        options=["PER", "PBV", "ROE", "Pertumbuhan Pendapatan", "Sentimen", "Dividen Yield"],
        default=["PER", "PBV", "ROE"]
    )
    
    if not metrics:
        st.warning("Pilih minimal satu metrik")
        return
    
    # Buat grafik untuk setiap metrik
    for metric in metrics:
        fig = px.bar(
            df,
            x="Ticker",
            y=metric,
            color="Sumber",
            title=f"Perbandingan {metric}",
            text=df[metric].apply(lambda x: f"{x:.2f}{'%' if metric != 'Sentimen' else ''}"),
            labels={"value": metric},
            color_discrete_map={
                "Portofolio Anda": "green",
                "Pasar Indonesia": "blue"
            }
        )
        
        fig.update_layout(
            yaxis_title=metric,
            xaxis_title="Saham"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Analisis komparatif
    st.subheader("Analisis Komparatif")
    
    # Temukan saham dengan nilai terbaik untuk setiap metrik
    best_stocks = {}
    for metric in metrics:
        if metric in ["PER", "PBV"]:  # Rendah lebih baik
            best = df.loc[df[metric].replace(0, np.nan).dropna().idxmin()]["Ticker"]
            best_stocks[metric] = {"saham": best, "nilai": df[metric].min()}
        else:  # Metrik lain: nilai tinggi lebih baik
            best = df.loc[df[metric].replace(0, np.nan).dropna().idxmax()]["Ticker"]
            best_stocks[metric] = {"saham": best, "nilai": df[metric].max()}
    
    # Tampilkan hasil analisis
    analysis_result = []
    for metric, data in best_stocks.items():
        analysis_result.append({
            "Metrik": metric,
            "Saham Terbaik": data["saham"],
            "Nilai": f"{data['nilai']:.2f}{'%' if metric != 'Sentimen' else ''}",
            "Kategori": "Portofolio Anda" if data["saham"] in selected_portfolio else "Pasar Indonesia"
        })
    
    # Tampilkan dengan warna
    def color_category(val):
        color = 'green' if val == 'Portofolio Anda' else 'blue'
        return f'background-color: {color}; color: white'
    
    st.dataframe(
        pd.DataFrame(analysis_result).style.applymap(
            color_category, 
            subset=['Kategori']
        ), 
        use_container_width=True
    )
    
    # Analisis komparatif
    st.subheader("Analisis Komparatif")
    
    # Temukan saham dengan nilai terbaik untuk setiap metrik
    best_stocks = {}
    for metric in metrics:
        if metric == "PER":  # PER rendah lebih baik
            best = df.loc[df[metric].idxmin()]["Ticker"]
            best_stocks[metric] = {"saham": best, "nilai": df[metric].min()}
        else:  # Metrik lain: nilai tinggi lebih baik
            best = df.loc[df[metric].idxmax()]["Ticker"]
            best_stocks[metric] = {"saham": best, "nilai": df[metric].max()}
    
    # Tampilkan hasil analisis
    analysis_result = []
    for metric, data in best_stocks.items():
        analysis_result.append({
            "Metrik": metric,
            "Saham Terbaik": data["saham"],
            "Nilai": f"{data['nilai']:.2f}{'%' if metric != 'Sentimen' else ''}"
        })
    
    st.dataframe(pd.DataFrame(analysis_result), use_container_width=True)

def get_stock_sentiment(ticker):
    """Hitung skor sentimen rata-rata untuk sebuah saham"""
    articles = get_news_from_yahoo(ticker)
    sentiment_scores = []
    
    for article in articles[:5]:  # Gunakan 5 berita terbaru
        text = f"{article['title']}. {article['description']}"
        sentiment = analyze_sentiment(text)
        sentiment_scores.append(sentiment['combined_score'])
    
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Fungsi untuk menghitung skor risiko portofolio - PERBAIKAN
def calculate_portfolio_risk_score(portfolio_df, api_key):
    st.subheader("📊 Skor Risiko Portofolio")
    
    if portfolio_df.empty:
        st.warning("Silakan upload portofolio Anda terlebih dahulu")
        return 0
    
    if not api_key:
        st.warning("Silakan masukkan API Key FMP di sidebar untuk fitur ini")
        return 0
    
    # PERBAIKAN: Pastikan data portfolio sudah diperbarui
    if 'Current Value' not in portfolio_df.columns:
        portfolio_df = update_portfolio_data(portfolio_df.copy())
        
        # Jika masih tidak ada kolom 'Current Value', gunakan kolom alternatif
        if 'Current Value' not in portfolio_df.columns:
            st.warning("Kolom 'Current Value' tidak ditemukan, menggunakan 'Avg Price' sebagai alternatif")
            portfolio_df['Current Value'] = portfolio_df['Lot Balance'] * portfolio_df['Avg Price']
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Faktor risiko
    risk_factors = []
    
    for i, (idx, row) in enumerate(portfolio_df.iterrows()):
        ticker = row['Ticker'].replace('.JK', '')
        status_text.text(f"Menganalisis risiko {ticker} ({i+1}/{len(portfolio_df)})...")
        
        try:
            fmp_data = get_fmp_data(ticker, api_key)
            if not fmp_data:
                continue
                
            ratios = fmp_data.get('ratios', {})
            profile = fmp_data.get('profile', {})
            quote = fmp_data.get('quote', {})
            
            # Faktor risiko
            beta = profile.get('beta', 1.0)
            vol = quote.get('change', 0)  # Volatilitas sederhana
            per = ratios.get('priceEarningsRatio', 15)
            der = ratios.get('debtEquityRatio', 0.5)
            size = profile.get('mktCap', 1000000000000)  # Market cap
            
            # Hitung skor risiko per saham (0-10, semakin tinggi semakin berisiko)
            risk_score = 0
            
            # Beta (1.0 = pasar, >1.0 lebih berisiko)
            if beta > 1.5:
                risk_score += 3
            elif beta > 1.2:
                risk_score += 2
            elif beta > 1.0:
                risk_score += 1
                
            # Volatilitas
            if abs(vol) > 5:
                risk_score += 3
            elif abs(vol) > 3:
                risk_score += 2
            elif abs(vol) > 1:
                risk_score += 1
                
            # PER tinggi -> lebih berisiko
            if per > 25:
                risk_score += 2
            elif per > 20:
                risk_score += 1
                
            # DER tinggi -> lebih berisiko
            if der > 2.0:
                risk_score += 3
            elif der > 1.5:
                risk_score += 2
            elif der > 1.0:
                risk_score += 1
                
            # Ukuran perusahaan (kecil -> lebih berisiko)
            if size < 500000000000:  # <500M
                risk_score += 3
            elif size < 1000000000000:  # <1T
                risk_score += 2
            elif size < 5000000000000:  # <5T
                risk_score += 1
                
            risk_factors.append({
                'Ticker': ticker,
                'Beta': beta,
                'Volatilitas': vol,
                'PER': per,
                'DER': der,
                'Market Cap': size,
                'Skor Risiko': min(10, risk_score)  # Batasi maks 10
            })
            
            progress_bar.progress((i+1)/len(portfolio_df))
            
        except Exception as e:
            st.error(f"Error menganalisis {ticker}: {str(e)}")
    
    status_text.text("Analisis risiko selesai!")
    
    if not risk_factors:
        st.warning("Tidak dapat menghitung skor risiko")
        return 0
    
    # Hitung skor risiko portofolio tertimbang
    total_value = portfolio_df['Current Value'].sum()
    portfolio_risk_score = 0
    
    for risk_factor in risk_factors:
        ticker = risk_factor['Ticker']
        # PERBAIKAN: Gunakan regex untuk pencocokan ticker yang lebih fleksibel
        mask = portfolio_df['Ticker'].str.contains(ticker, regex=False)
        stock_value = portfolio_df.loc[mask, 'Current Value'].sum()
        
        if total_value > 0:
            weight = stock_value / total_value
        else:
            weight = 0
            
        portfolio_risk_score += risk_factor['Skor Risiko'] * weight
    
    # Tampilkan hasil
    st.metric("Skor Risiko Portofolio", f"{portfolio_risk_score:.1f}/10.0", 
             "Rendah" if portfolio_risk_score < 3 else 
             "Sedang" if portfolio_risk_score < 6 else 
             "Tinggi" if portfolio_risk_score < 8 else "Sangat Tinggi")
    
    # Interpretasi
    if portfolio_risk_score < 3:
        st.success("Portofolio Anda memiliki risiko rendah. Cocok untuk investor konservatif.")
    elif portfolio_risk_score < 6:
        st.info("Portofolio Anda memiliki risiko sedang. Seimbang antara risiko dan potensi return.")
    elif portfolio_risk_score < 8:
        st.warning("Portofolio Anda memiliki risiko tinggi. Cocok untuk investor agresif dengan toleransi risiko tinggi.")
    else:
        st.error("Portofolio Anda memiliki risiko sangat tinggi. Pertimbangkan untuk diversifikasi lebih banyak.")
    
    # Tampilkan detail saham
    st.subheader("Detail Risiko Saham")
    df_risk = pd.DataFrame(risk_factors)
    st.dataframe(df_risk.style.format({
        'Beta': '{:.2f}',
        'Volatilitas': '{:.2f}%',
        'PER': '{:.2f}',
        'DER': '{:.2f}',
        'Market Cap': 'Rp {:,.0f}'
    }).background_gradient(subset=['Skor Risiko'], cmap='YlOrRd'), use_container_width=True)
    
    return portfolio_risk_score

# Sidebar menu - hanya satu blok sidebar
st.sidebar.title("📋 Menu Analisis")
st.sidebar.header("Konfigurasi API")
api_key = get_fmp_api_key()
news_api_key = get_news_api_key()

st.sidebar.header("Portfolio")
uploaded_file = st.sidebar.file_uploader("Upload Portfolio", type=["csv", "xlsx"])

portfolio_df = pd.DataFrame()
if uploaded_file:
    portfolio_df = process_uploaded_file(uploaded_file)
    if not portfolio_df.empty:
        st.sidebar.success("File portfolio berhasil diupload!")
        st.sidebar.dataframe(portfolio_df[['Stock', 'Ticker']])
        
# Update menu options
menu_options = [
    "Dashboard Portfolio",
    "Analisis DCA",
    "Prediksi Harga Saham",
    "Valuasi Saham",
    "Tracking Modal",
    "Rekomendasi Pembelian",
    "Market News & Sentiment",
    "Smart Assistant & Rekomendasi AI",
    "Komparasi Saham"  # Fitur baru ditambahkan
]
selected_menu = st.sidebar.selectbox("Pilih Fitur:", menu_options)

# Main content area
st.title("📈 Stock Analysis Toolkit Pro+")

# Tambahkan tombol hamburger untuk mobile
st.markdown("""
    <style>
        /* Tombol hamburger untuk mobile */
        .hamburger {
            display: none;
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 1001;
            background: #1e3a8a;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 1.5rem;
        }
        
        @media (max-width: 768px) {
            .hamburger {
                display: block;
            }
            
            /* Sembunyikan sidebar default */
            div[data-testid="stSidebar"] {
                transform: translateX(-100%);
                transition: transform 0.3s ease;
            }
            
            div[data-testid="stSidebar"].active {
                transform: translateX(0);
            }
        }
    </style>
    
    <button class="hamburger" onclick="toggleSidebar()">☰</button>
    
    <script>
        function toggleSidebar() {
            const sidebar = document.querySelector('div[data-testid="stSidebar"]');
            sidebar.classList.toggle('active');
        }
    </script>
""", unsafe_allow_html=True)

if selected_menu == "Dashboard Portfolio":
    if not portfolio_df.empty:
        portfolio_df = dca_analysis(portfolio_df)
    else:
        st.info("Silakan upload file portfolio untuk melihat dashboard")

elif selected_menu == "Analisis DCA":
    if not portfolio_df.empty:
        portfolio_df = dca_analysis(portfolio_df)
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")

elif selected_menu == "Prediksi Harga Saham":
    if not portfolio_df.empty:
        selected_ticker = st.selectbox("Pilih Saham", portfolio_df['Ticker'].tolist())
        stock_prediction(selected_ticker)
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")

elif selected_menu == "Valuasi Saham":
    if not portfolio_df.empty and api_key:
        selected_ticker = st.selectbox("Pilih Saham", portfolio_df['Ticker'].tolist())
        clean_ticker = selected_ticker.replace('.JK', '')
        stock_valuation(clean_ticker, api_key)
    elif not api_key:
        st.warning("Silakan masukkan API Key FMP di sidebar")
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")

elif selected_menu == "Tracking Modal":
    capital_tracking()

elif selected_menu == "Rekomendasi Pembelian":
    if not portfolio_df.empty and api_key:
        portfolio_df = investment_simulation(portfolio_df, api_key)
    elif not api_key:
        st.warning("Silakan masukkan API Key FMP di sidebar")
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")
        
elif selected_menu == "Market News & Sentiment":
    display_news_feed()
    
# Menu Smart Assistant & Rekomendasi AI yang sudah diperbaiki
elif selected_menu == "Smart Assistant & Rekomendasi AI":
    st.header("🤖 Smart Assistant & Rekomendasi AI")
    
    tab1, tab2, tab3 = st.tabs([
        "Saham Undervalued", 
        "Rekomendasi Diversifikasi", 
        "Skor Risiko Portofolio"
    ])
    
    with tab1:
        st.subheader("Rekomendasi Saham Undervalued")
        st.info("Berikut rekomendasi saham yang dianggap undervalued berdasarkan analisis fundamental:")
        get_undervalued_recommendations(api_key)
    
    with tab2:
        st.subheader("Rekomendasi Diversifikasi Portofolio")
        st.info("Dapatkan rekomendasi alokasi portofolio berdasarkan profil risiko Anda:")
        
        # Dapatkan profil risiko
        risk_profile = get_risk_profile()
        
        if risk_profile and not portfolio_df.empty:
            updated_df = update_portfolio_data(portfolio_df.copy())
            get_diversification_recommendation(updated_df, risk_profile)
    
    with tab3:
        st.subheader("Analisis Risiko Portofolio")
        st.info("Skor risiko portofolio Anda berdasarkan karakteristik saham:")
        
        if not portfolio_df.empty and api_key:
            updated_df = update_portfolio_data(portfolio_df.copy())
            calculate_portfolio_risk_score(updated_df, api_key)

# Tambahkan else untuk menangani kasus yang tidak terduga
elif selected_menu == "Komparasi Saham":
    if api_key:
        stock_comparison(api_key, portfolio_df)
    else:
        st.warning("Silakan masukkan API Key FMP di sidebar")
