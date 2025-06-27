import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

# Konfigurasi awal
st.set_page_config(layout="wide", page_title="Analisis Portofolio Saham")

# Fungsi untuk memuat data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Bersihkan data
        df = df.dropna(subset=['Stock'])
        # Konversi harga ke float
        df['Avg Price'] = df['Avg Price'].str.replace('Rp ', '').str.replace(',', '').astype(float)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

# Fungsi untuk mendapatkan data harga saham
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except:
        return pd.DataFrame()

# Fungsi untuk analisis DCA
def dca_analysis(df):
    results = []
    for _, row in df.iterrows():
        ticker = row['Ticker']
        avg_price = row['Avg Price']
        current_price = get_current_price(ticker)
        
        if not np.isnan(current_price):
            # Simulasi DCA
            dca_values = []
            for months in [6, 12, 24]:
                simulated_price = simulate_dca(ticker, months)
                dca_values.append({
                    'period': f"{months} bulan",
                    'simulated_price': simulated_price
                })
            
            results.append({
                'Stock': row['Stock'],
                'Ticker': ticker,
                'Avg Price': avg_price,
                'Current Price': current_price,
                'DCA Analysis': dca_values,
                'Performance': (current_price - avg_price) / avg_price * 100
            })
    
    return pd.DataFrame(results)

# Fungsi untuk mendapatkan harga terkini
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else np.nan
    except:
        return np.nan

# Fungsi simulasi DCA
def simulate_dca(ticker, months):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months*30)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if not hist.empty:
            return hist['Close'].mean()
        return np.nan
    except:
        return np.nan

# Fungsi untuk mendapatkan sentimen berita
def get_news_sentiment(ticker):
    try:
        # Ganti dengan implementasi API berita sebenarnya
        # Contoh placeholder
        time.sleep(0.5)  # Delay untuk simulasi
        
        # Simulasi sentimen acak
        sentiments = ['Positif', 'Netral', 'Negatif']
        return np.random.choice(sentiments, p=[0.3, 0.5, 0.2])
    except:
        return "Tidak tersedia"

# Fungsi rekomendasi saham
def generate_recommendations():
    # Data saham rekomendasi (contoh statis)
    return pd.DataFrame({
        'Saham': ['BBCA', 'BBRI', 'TLKM', 'ASII', 'GGRM'],
        'Harga (Rp)': [8500, 4850, 3650, 6150, 4250],
        'Dividen Yield (%)': [3.2, 4.1, 3.8, 2.9, 4.5],
        'Rekomendasi': ['BELI', 'BELI', 'TAMBAH', 'HOLD', 'BELI'],
        'Keterangan': [
            'Bluechip dengan dividen stabil',
            'Dividen tinggi dengan fundamental kuat',
            'Prospek pertumbuhan baik',
            'Harga stabil, tunggu koreksi',
            'Dividen tinggi dengan valuasi menarik'
        ]
    })

# Fungsi analisis rekomendasi portofolio
def analyze_portfolio(df):
    recommendations = []
    for _, row in df.iterrows():
        ticker = row['Ticker']
        avg_price = row['Avg Price']
        current_price = get_current_price(ticker)
        
        if not np.isnan(current_price):
            performance = (current_price - avg_price) / avg_price * 100
            
            if performance > 25:
                rec = 'JUAL'
                reason = f'Kenaikan signifikan ({performance:.2f}%)'
            elif performance < -15:
                rec = 'TAMBAH'
                reason = f'Potensi averaging down ({performance:.2f}%)'
            elif performance > 10:
                rec = 'HOLD'
                reason = f'Kinerja baik ({performance:.2f}%)'
            else:
                rec = 'HOLD'
                reason = 'Performa wajar'
            
            recommendations.append({
                'Saham': row['Stock'],
                'Ticker': ticker,
                'Rekomendasi': rec,
                'Alasan': reason,
                'Harga Avg (Rp)': avg_price,
                'Harga Sekarang (Rp)': current_price,
                'Performa (%)': performance
            })
    
    return pd.DataFrame(recommendations)

# Tampilan Streamlit
st.title("📈 Analisis Portofolio Saham")

# Upload file
uploaded_file = st.file_uploader("Unggah file portofolio saham (CSV)", type="csv")
df = pd.DataFrame()

if uploaded_file:
    df = load_data(uploaded_file)
    
    if not df.empty:
        st.success("Data berhasil dimuat!")
        st.subheader("Portofolio Saat Ini")
        st.dataframe(df.style.format({'Avg Price': 'Rp {:,.0f}'}), height=300)
        
        # Analisis DCA
        st.subheader("🔍 Analisis Dollar Cost Averaging (DCA)")
        dca_df = dca_analysis(df)
        
        if not dca_df.empty:
            # Tampilkan hasil DCA
            for _, row in dca_df.iterrows():
                with st.expander(f"{row['Stock']} ({row['Ticker']})"):
                    st.write(f"**Harga Rata-rata:** Rp {row['Avg Price']:,.0f}")
                    st.write(f"**Harga Sekarang:** Rp {row['Current Price']:,.0f}")
                    st.write(f"**Performa:** {row['Performance']:.2f}%")
                    
                    # Grafik performa
                    fig = px.line(
                        get_stock_data(row['Ticker']).reset_index(),
                        x='Date',
                        y='Close',
                        title=f"Performa {row['Stock']} ({row['Ticker']})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analisis DCA
                    st.subheader("Simulasi DCA")
                    for dca in row['DCA Analysis']:
                        st.write(f"- Rata-rata {dca['period']}: Rp {dca['simulated_price']:,.0f}")
        
        # Sentimen Berita
        st.subheader("📰 Sentimen Berita Saham")
        sentiment_df = pd.DataFrame({
            'Saham': df['Stock'],
            'Sentimen': [get_news_sentiment(t) for t in df['Ticker']]
        })
        st.dataframe(sentiment_df)
        
        # Rekomendasi Portofolio
        st.subheader("🚦 Rekomendasi Portofolio")
        rec_df = analyze_portfolio(df)
        if not rec_df.empty:
            st.dataframe(rec_df.style.applymap(
                lambda x: 'background-color: lightgreen' if x == 'TAMBAH' else 
                         ('background-color: salmon' if x == 'JUAL' else ''),
                subset=['Rekomendasi']
            ))
        
        # Saham Rekomendasi
        st.subheader("💡 Saham Rekomendasi")
        st.write("Saham dengan potensi dividen tinggi dan valuasi menarik:")
        st.dataframe(generate_recommendations())
        
        # Ringkasan Portofolio
        st.subheader("📊 Ringkasan Portofolio")
        if not rec_df.empty:
            total_investment = (df['Lot Balance'] * df['Avg Price']).sum()
            current_value = (df['Lot Balance'] * rec_df['Harga Sekarang (Rp)']).sum()
            performance = (current_value - total_investment) / total_investment * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investasi", f"Rp {total_investment:,.0f}")
            col2.metric("Nilai Sekarang", f"Rp {current_value:,.0f}")
            col3.metric("Performa", f"{performance:.2f}%", 
                       delta_color="inverse" if performance < 0 else "normal")
            
            # Grafik alokasi
            allocation = df.copy()
            allocation['Nilai'] = allocation['Lot Balance'] * rec_df['Harga Sekarang (Rp)']
            fig = px.pie(
                allocation, 
                names='Stock', 
                values='Nilai',
                title='Alokasi Portofolio'
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Silakan unggah file portofolio saham dalam format CSV")

# Catatan kaki
st.markdown("---")
st.caption("© 2023 Tools Analisis Saham | Data harga saham bersumber dari Yahoo Finance")
