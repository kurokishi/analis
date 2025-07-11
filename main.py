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
            df = pm.analyze_dca()
            summary = pm.get_summary_metrics()

            st.subheader("📊 Ringkasan Portfolio")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investasi", f"Rp {summary['total_investment']:,.0f}")
            col2.metric("Nilai Saat Ini", f"Rp {summary['total_current_value']:,.0f}")
            col3.metric("Profit/Loss", f"Rp {summary['total_profit']:,.0f}", 
                        f"{summary['total_profit_percent']:+.2f}%")

            st.dataframe(df.style.format({
                'Avg Price': 'Rp {:,.0f}',
                'Current Price': 'Rp {:,.0f}',
                'Current Value': 'Rp {:,.0f}',
                'Profit/Loss': 'Rp {:,.0f}',
                'Profit/Loss %': '{:+.2f}%'
            }), use_container_width=True)

            # Pie chart: Komposisi berdasarkan nilai saat ini
            st.subheader("📈 Komposisi Portfolio Saat Ini")
            fig = px.pie(df, names='Ticker', values='Current Value', title='Distribusi Saham')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            # Bar chart: Profit/Loss per saham
            st.subheader("📊 Profit/Loss per Saham")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df['Ticker'],
                y=df['Profit/Loss'],
                marker_color=['green' if x >= 0 else 'red' for x in df['Profit/Loss']],
                text=df['Profit/Loss'].apply(lambda x: f"Rp {x:,.0f}"),
                textposition="auto"
            ))
            fig2.update_layout(title="Keuntungan/Kerugian Tiap Saham", xaxis_title="Saham", yaxis_title="Rp")
            st.plotly_chart(fig2, use_container_width=True)

        elif menu == "Rekomendasi Pembelian":
            if not fmp_api_key:
                st.warning("Masukkan API Key FMP di sidebar terlebih dahulu")
            else:
                fmp = FMPClient(fmp_api_key)
                va = ValuationAnalyzer(fmp)
                amount = st.number_input("Modal Investasi (Rp)", value=500000, step=100000, min_value=100000)
                result_df = va.simulate_allocation(df, amount)

                st.subheader("📌 Rekomendasi Alokasi")
                st.dataframe(result_df[[
                    'Ticker', 'Valuation Score', 'Current Price', 'Additional Shares', 'Additional Investment'
                ]].style.format({
                    'Current Price': 'Rp {:,.0f}',
                    'Additional Investment': 'Rp {:,.0f}'
                }), use_container_width=True)

                fig = px.pie(result_df[result_df['Additional Shares'] > 0], 
                             names='Ticker', values='Additional Investment', 
                             title='Distribusi Alokasi Modal Baru')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

        elif menu == "Prediksi Saham":
            selected_ticker = st.selectbox("Pilih Saham", df['Ticker'].tolist())
            predictor = StockPredictor(selected_ticker)
            trend, pred_price = predictor.predict_trend()

            st.subheader(f"📈 Prediksi Saham {selected_ticker}")
            last_close = predictor.data['Close'].iloc[-1] if not predictor.data.empty else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Terakhir", f"Rp {last_close:,.0f}")
            col2.metric("Prediksi Harga", f"Rp {pred_price:,.0f}")
            col3.metric("Trend", trend)

            # Tambahkan chart harga historis dengan MA
            st.subheader("Grafik Harga & MA")
            fig = go.Figure()
            dfp = predictor.data
            fig.add_trace(go.Scatter(x=dfp.index, y=dfp['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=dfp.index, y=dfp['MA20'], name='MA20'))
            fig.add_trace(go.Scatter(x=dfp.index, y=dfp['MA50'], name='MA50'))
            fig.update_layout(title=f'Harga Historis {selected_ticker}', xaxis_title='Tanggal', yaxis_title='Harga')
            st.plotly_chart(fig, use_container_width=True)

        elif menu == "Berita & Sentimen":
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
