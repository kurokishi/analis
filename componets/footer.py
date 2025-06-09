# components/footer.py
import streamlit as st

def render_footer():
    st.markdown("""
        ---
        <div style="text-align: center; color: #7f8c8d;">
        <p>© 2023 Saham Analyzer Pro | Data saham dari Yahoo Finance</p>
        <p>Disclaimer: Analisis ini bukan rekomendasi investasi. Lakukan riset mandiri sebelum berinvestasi.</p>
        </div>
    """, unsafe_allow_html=True)
