import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
import logging
import traceback
from portfolio_manager import PortfolioManager
from fmp_client import FMPClient
from valuation_analyzer import ValuationAnalyzer
from stock_predictor import StockPredictor
from news_analyzer import NewsAnalyzer

# FIX: Tambahkan perbaikan untuk issue cache yfinance
cache_dir = Path.home() / ".cache" / "py-yfinance"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['YFINANCE_CACHE_DIR'] = str(cache_dir)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Stock Analysis Pro", layout="wide")
st.title("📈 Stock Analysis Toolkit - OOP Version")

# Sidebar - API Key dan Upload File
st.sidebar.header("🔑 Konfigurasi")
fmp_api_key = st.sidebar.text_input("API Key FMP", type="password")
news_ticker = st.sidebar.text_input("Ticker untuk Berita", "AAPL")

uploaded_file = st.sidebar.file_uploader("Upload Portfolio (.csv / .xlsx)", type=["csv", "xlsx"])

portfolio_df = pd.DataFrame()
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            portfolio_df = pd.read_csv(uploaded_file)
        else:
            portfolio_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal memproses file: {e}")

# Menu
menu = st.sidebar.selectbox("📌 Pilih Fitur", [
    "Dashboard Portfolio", 
    "Rekomendasi Pembelian", 
    "Prediksi Saham",
    "Berita & Sentimen"
])

# Proses jika data tersedia
if not portfolio_df.empty:
    try:
        pm = PortfolioManager(portfolio_df)
        pm.update_realtime_prices()
        df = pm.get_dataframe()

        if menu == "Dashboard Portfolio":
            # [Kode dashboard tetap sama...]

           elif menu == "Rekomendasi Pembelian":
            # [Kode rekomendasi tetap sama...]

           elif menu == "Prediksi Saham":
            # [Kode prediksi tetap sama...]

           elif menu == "Berita & Sentimen":
            # PERBAIKAN BAGIAN BERITA
                try:
                    analyzer = NewsAnalyzer(language='id')
                       with st.spinner('Mengambil berita terbaru...'):
                            articles = analyzer.fetch_news(news_ticker, max_articles=10)
                
                       if not articles:
                            st.warning("Tidak ditemukan berita untuk saham ini.")
                       else:
                            analyzed = analyzer.analyze_articles(articles)
                            st.subheader(f"📰 Berita Terkait {news_ticker}")
                    
                            # Ringkasan sentimen
                            sentiment_summary = analyzer.summarize_sentiment(analyzed)
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Berita", sentiment_summary['total_articles'])
                            col2.metric("Positif", sentiment_summary['positive'])
                            col3.metric("Netral", sentiment_summary['neutral'])
                            col4.metric("Negatif", sentiment_summary['negative'])
                    
                    for a in analyzed:
                        sentiment = a.get('sentiment_label', 'Netral')
                        sentiment_score = a['sentiment']['combined_score']
                        color = "green" if sentiment == "Positif" else ("red" if sentiment == "Negatif" else "gray")
                        
                        st.markdown(f"**{a['title']}**")
                        st.markdown(f"*{a['source']} - {a['published_at']}*")
                        st.markdown(f"{a['description']}")
                        st.markdown(f"**Sentimen**: :{color}[{sentiment} ({sentiment_score:.2f})]")
                        st.markdown(f"[Baca Selengkapnya]({a['url']})")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Gagal mengambil berita: {str(e)}")
                logger.error(traceback.format_exc())
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memproses portfolio: {str(e)}")
        logger.error(traceback.format_exc())
else:
    st.info("Silakan upload file portfolio terlebih dahulu untuk mengakses fitur-fitur.")
