# app.py (Entry Point - Refactored OOP & Modular)
import streamlit as st
from datetime import datetime

from core.analyzer import SahamAnalyzer
from core.formatter import format_currency, format_percent, get_color
from components.header import render_header, render_sidebar
from components.footer import render_footer
from components.widgets import (
    render_summary_tab, render_technical_tab,
    render_fundamental_tab, render_projection_tab,
    render_prediction_tab
)

# Konfigurasi halaman
st.set_page_config(page_title="Saham Analyzer Pro", layout="wide", page_icon="📈")

# Header dan sidebar
render_header()
ticker, days, show_rsi, show_macd, show_bbands, growth_rate, pe_target = render_sidebar()

# Inisialisasi dan ambil data
if ticker:
    try:
        with st.spinner(f"Mengambil data untuk {ticker}..."):
            analyzer = SahamAnalyzer(ticker, days)
            analyzer.fetch_data()

        if analyzer.df is not None and analyzer.info is not None:
            analyzer.process_fundamental()
            analyzer.apply_technical_indicators(rsi=show_rsi, macd=show_macd, bbands=show_bbands)
            analyzer.make_projections(growth_rate=growth_rate, pe_target=pe_target)
            rekom, skor, alasan = analyzer.get_recommendation()

            # Tabs UI
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Ringkasan", "📈 Teknikal", "📉 Fundamental", "🔮 Proyeksi", "🤖 AI Prediksi"])

            with tab1:
                render_summary_tab(analyzer, rekom, skor, alasan)

            with tab2:
                render_technical_tab(analyzer)

            with tab3:
                render_fundamental_tab(analyzer)

            with tab4:
                render_projection_tab(analyzer, growth_rate, pe_target)

            with tab5:
                render_prediction_tab(analyzer)

        else:
            st.error("Gagal mengambil data saham. Pastikan kode saham benar dan coba lagi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.info("Pastikan kode saham menggunakan format yang benar (contoh: BBCA.JK) dan coba lagi.")

else:
    st.info("Masukkan kode saham di sidebar untuk memulai analisis.")

# Footer
render_footer()
