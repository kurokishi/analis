# components/widgets.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from core.formatter import format_currency, format_percent, get_color
import io
from fpdf import FPDF
import xlsxwriter

# Fungsi tab ringkasan
def render_summary_tab(analyzer, rekom, skor, alasan):
    df, info, fundamental = analyzer.df, analyzer.info, analyzer.fundamental
    st.header(f"Ringkasan Saham {analyzer.ticker}")

    col1, col2, col3, col4 = st.columns(4)
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_close
    pct_change = (price_change / prev_close) * 100

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Harga Terakhir</h4>
            <h2>{format_currency(current_price)}</h2>
            <p class="{get_color(price_change)}">
                {format_currency(price_change)} ({format_percent(pct_change / 100)})
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Market Cap</h4>
            <h2>{format_currency(info.get("market_cap"))}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>P/E Ratio</h4>
            <h2>{info.get("pe_ratio") or 'N/A'}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Dividend Yield</h4>
            <h2>{format_percent(info.get("dividend_yield"))}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Ringkasan fundamental
    st.subheader("Ringkasan Fundamental")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Valuasi Saham**")
        st.markdown(f"```\n{fundamental['Valuation']}\n```")
    with cols[1]:
        st.markdown("**Profitabilitas**")
        st.markdown(f"- ROE: {fundamental['ROE']}")
        st.markdown(f"- ROA: {fundamental.get('ROA', 'N/A')}")
        st.markdown(f"- Margin Laba: {fundamental.get('Profit Margin', 'N/A')}")
    with cols[2]:
        st.markdown("**Rasio Keuangan**")
        st.markdown(f"- P/E: {fundamental['PE Ratio']}")
        st.markdown(f"- P/B: {fundamental['PB Ratio']}")
        st.markdown(f"- DER: {fundamental.get('DER', 'N/A')}")

    # Tombol download ringkasan
    st.subheader("📄 Unduh Ringkasan")
    summary_text = f"""
Ticker: {analyzer.ticker}
Harga Terakhir: {format_currency(current_price)} ({format_percent(pct_change / 100)})
Market Cap: {format_currency(info.get("market_cap"))}
P/E: {info.get("pe_ratio")}, Yield: {format_percent(info.get("dividend_yield"))}

Valuasi: {fundamental['Valuation']}
ROE: {fundamental['ROE']}, ROA: {fundamental.get('ROA', 'N/A')}
Profit Margin: {fundamental.get('Profit Margin', 'N/A')}

Rekomendasi: {rekom} (Skor: {skor}/10)
Alasan:
"""
    for i, r in enumerate(alasan):
        summary_text += f"- {r}\n"

    # TXT
    buffer_txt = io.BytesIO()
    buffer_txt.write(summary_text.encode())
    buffer_txt.seek(0)
    st.download_button("📥 Download TXT", data=buffer_txt, file_name=f"ringkasan_{analyzer.ticker}.txt", mime="text/plain")

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for line in summary_text.splitlines():
        pdf.cell(200, 6, txt=line, ln=True)
    buffer_pdf = io.BytesIO()
    pdf.output(buffer_pdf)
    buffer_pdf.seek(0)
    st.download_button("📥 Download PDF", data=buffer_pdf, file_name=f"ringkasan_{analyzer.ticker}.pdf", mime="application/pdf")

    # Excel
    buffer_xlsx = io.BytesIO()
    with pd.ExcelWriter(buffer_xlsx, engine="xlsxwriter") as writer:
        summary_df = pd.DataFrame.from_dict({
            'Ticker': [analyzer.ticker],
            'Harga': [format_currency(current_price)],
            'Market Cap': [format_currency(info.get("market_cap"))],
            'P/E': [info.get("pe_ratio")],
            'Dividend Yield': [format_percent(info.get("dividend_yield"))],
            'Valuasi': [fundamental['Valuation']],
            'ROE': [fundamental['ROE']],
            'ROA': [fundamental.get('ROA', 'N/A')],
            'Profit Margin': [fundamental.get('Profit Margin', 'N/A')],
            'Rekomendasi': [rekom],
            'Skor': [skor]
        }, orient='columns')
        summary_df.to_excel(writer, index=False, sheet_name="Ringkasan")
    buffer_xlsx.seek(0)
    st.download_button("📥 Download Excel", data=buffer_xlsx, file_name=f"ringkasan_{analyzer.ticker}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
