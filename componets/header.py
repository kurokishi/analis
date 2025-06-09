# components/header.py
import streamlit as st
import json
import os

FAVORITE_FILE = "favorite_tickers.json"

def load_favorites():
    if os.path.exists(FAVORITE_FILE):
        with open(FAVORITE_FILE, "r") as f:
            return json.load(f)
    return []

def save_favorite(ticker):
    favs = load_favorites()
    if ticker not in favs:
        favs.append(ticker)
        with open(FAVORITE_FILE, "w") as f:
            json.dump(favs, f)

def render_header():
    st.markdown("""
        <style>
        .main {background-color: #f5f5f5;}
        .stButton>button {border-radius: 8px;}
        .stTextInput>div>div>input {border-radius: 8px;}
        .metric-card {border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        .positive {color: #2ecc71;}
        .negative {color: #e74c3c;}
        .header {color: #3498db;}
        </style>
        <div style="background-color: #3498db; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; text-align: center;">Analisis Fundamental & Teknikal Saham Indonesia</h3>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Pengaturan Analisis")

        favorites = load_favorites()
        if favorites:
            st.markdown("**Ticker Favorit Anda:**")
            selected = st.selectbox("Pilih dari favorit", favorites)
        else:
            selected = "BBCA.JK"

        ticker = st.text_input("Masukkan kode saham (contoh: BBCA.JK)", value=selected)

        if st.button("⭐ Simpan ke Favorit"):
            save_favorite(ticker)
            st.success(f"Ticker {ticker} disimpan ke favorit!")

        time_period = st.selectbox("Periode Data Historis", ["3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun", "5 Tahun"], index=2)
        period_map = {"3 Bulan": 90, "6 Bulan": 180, "1 Tahun": 365, "2 Tahun": 730, "5 Tahun": 1825}
        days = period_map[time_period]

        st.subheader("Indikator Teknikal")
        show_rsi = st.checkbox("RSI (14 hari)", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_bbands = st.checkbox("Bollinger Bands", value=False)

        st.markdown("---")
        st.markdown("""**ℹ️ Tentang Aplikasi**
        Aplikasi ini memberikan analisis saham otomatis meliputi:
        - Analisis fundamental
        - Analisis teknikal
        - Proyeksi harga
        - Rekomendasi sistem
        """)

        growth_rate = 0.10
        pe_target = 15

    return ticker, days, show_rsi, show_macd, show_bbands, growth_rate, pe_target
