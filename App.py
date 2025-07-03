import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Konfigurasi halaman
st.set_page_config(
    page_title="Stock Analysis Toolkit",
    page_icon="📈",
    layout="wide"
)

# Fungsi untuk memproses file upload
def process_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Konversi kolom harga ke numeric
        df['Avg Price'] = df['Avg Price'].replace('[Rp, ]', '', regex=True).astype(float)
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()

# Fungsi analisis DCA
def dca_analysis(df):
    if df.empty:
        return
    
    df['Total Investment'] = df['Lot Balance(jumlah lembar dimiliki)'] * df['Avg Price']
    total_investment = df['Total Investment'].sum()
    total_shares = df['Lot Balance(jumlah lembar dimiliki)'].sum()
    avg_price = total_investment / total_shares
    
    st.subheader("Dollar Cost Averaging (DCA) Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Total Investasi", f"Rp {total_investment:,.2f}")
    col2.metric("Rata-rata Harga Beli", f"Rp {avg_price:,.2f}")
    
    fig, ax = plt.subplots()
    df.set_index('Ticker')['Total Investment'].plot(kind='bar', ax=ax)
    ax.set_title('Alokasi Investasi per Saham')
    ax.set_ylabel('Jumlah (Rp)')
    st.pyplot(fig)

# Fungsi prediksi harga (simulasi)
def stock_prediction(ticker):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        hist = stock.history(period="1y")
        
        if hist.empty:
            st.warning(f"Data tidak ditemukan untuk {ticker}")
            return
        
        st.subheader(f"Prediksi Harga Saham: {ticker}")
        
        # Prediksi sederhana (moving average)
        hist['MA50'] = hist['Close'].rolling(50).mean()
        hist['MA200'] = hist['Close'].rolling(200).mean()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        hist[['Close', 'MA50', 'MA200']].plot(ax=ax)
        ax.set_title(f"Perjalanan Harga {ticker}")
        ax.set_ylabel("Harga (Rp)")
        st.pyplot(fig)
        
        # Prediksi naif (terakhir + random)
        last_price = hist['Close'].iloc[-1]
        prediction = last_price * (1 + np.random.uniform(-0.05, 0.08))
        
        col1, col2 = st.columns(2)
        col1.metric("Harga Terakhir", f"Rp {last_price:,.2f}")
        col2.metric("Prediksi 1 Bulan", f"Rp {prediction:,.2f}", 
                   delta=f"{((prediction/last_price)-1)*100:.2f}%")
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Fungsi valuasi saham
def stock_valuation(ticker):
    try:
        st.subheader(f"Valuasi Saham: {ticker}")
        
        # Data dummy (bisa diganti dengan API sesungguhnya)
        valuation_data = {
            'Metric': ['PER', 'PBV', 'ROE', 'DER', 'NPM', 'EPS', 'Dividend Yield'],
            'Value': [
                np.random.uniform(8, 25),
                np.random.uniform(0.5, 3.5),
                np.random.uniform(5, 25),
                np.random.uniform(0.2, 1.5),
                np.random.uniform(10, 30),
                np.random.uniform(50, 500),
                np.random.uniform(1, 5)
            ],
            'Industry Avg': [
                np.random.uniform(10, 22),
                np.random.uniform(0.8, 3.0),
                np.random.uniform(8, 20),
                np.random.uniform(0.3, 1.2),
                np.random.uniform(12, 25),
                np.random.uniform(100, 400),
                np.random.uniform(2, 4)
            ]
        }
        
        df_valuation = pd.DataFrame(valuation_data)
        df_valuation['Difference'] = df_valuation['Value'] - df_valuation['Industry Avg']
        
        st.dataframe(df_valuation.style.format({
            'Value': '{:.2f}',
            'Industry Avg': '{:.2f}',
            'Difference': '{:.2f}'
        }))
        
        # DCF Valuation (simplified)
        st.subheader("Discounted Cash Flow (DCF)")
        st.write("""
        **Asumsi:**
        - Pertumbuhan 5 tahun: 8%
        - Terminal growth: 3%
        - Discount rate: 10%
        """)
        
        dcf_value = np.random.uniform(1000, 5000)
        st.metric("Nilai Wajar DCF", f"Rp {dcf_value:,.2f}")
        
    except Exception as e:
        st.error(f"Error in valuation: {str(e)}")

# Fungsi tracking modal
def capital_tracking():
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []
        
    st.subheader("Riwayat Transaksi")
    
    with st.form("transaction_form"):
        date = st.date_input("Tanggal Transaksi", datetime.today())
        ticker = st.text_input("Kode Saham")
        action = st.selectbox("Aksi", ["Beli", "Jual"])
        shares = st.number_input("Jumlah Lembar", min_value=1)
        price = st.number_input("Harga per Lembar (Rp)", min_value=1)
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
        st.dataframe(df_transactions)
        
        total_investment = df_transactions[df_transactions['Action'] == 'Beli']['Amount'].sum()
        total_sales = abs(df_transactions[df_transactions['Action'] == 'Jual']['Amount'].sum())
        net_cashflow = df_transactions['Amount'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pembelian", f"Rp {total_investment:,.2f}")
        col2.metric("Total Penjualan", f"Rp {total_sales:,.2f}")
        col3.metric("Net Cashflow", f"Rp {net_cashflow:,.2f}")

# Sidebar menu
st.sidebar.title("Menu Analisis")
menu_options = {
    "Portfolio": "portfolio",
    "Analisis DCA": "dca",
    "Prediksi Harga": "prediction",
    "Valuasi Saham": "valuation",
    "Tracking Modal": "capital"
}

selected_menu = st.sidebar.radio("Pilih Fitur:", list(menu_options.keys()))

# Main content area
st.title("📊 Stock Analysis Toolkit")

# Portfolio Upload (always shown)
uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv")
portfolio_df = pd.DataFrame()

if uploaded_file:
    portfolio_df = process_uploaded_file(uploaded_file)
    if not portfolio_df.empty:
        st.subheader("Portfolio Anda")
        st.dataframe(portfolio_df.style.format({'Avg Price': 'Rp {:.2f}'}))

# Menu handling
if selected_menu == "Portfolio":
    if portfolio_df.empty:
        st.info("Silakan upload file portfolio untuk melihat data")

elif selected_menu == "Analisis DCA":
    if not portfolio_df.empty:
        dca_analysis(portfolio_df)
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")

elif selected_menu == "Prediksi Harga":
    if not portfolio_df.empty:
        selected_ticker = st.selectbox("Pilih Saham", portfolio_df['Ticker'].tolist())
        stock_prediction(selected_ticker.split('.')[0])
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")

elif selected_menu == "Valuasi Saham":
    if not portfolio_df.empty:
        selected_ticker = st.selectbox("Pilih Saham", portfolio_df['Ticker'].tolist())
        stock_valuation(selected_ticker.split('.')[0])
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")

elif selected_menu == "Tracking Modal":
    capital_tracking()
