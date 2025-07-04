import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Stock Analysis Toolkit Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk mendapatkan API key - PERBAIKAN DI SINI
def get_fmp_api_key():
    if 'fmp_api_key' not in st.session_state:
        with st.sidebar:
            st.subheader("FinancialModelingPrep API")
            api_key = st.text_input("Masukkan API Key FMP", type="password")
            if st.button("Simpan API Key"):
                st.session_state.fmp_api_key = api_key
                st.success("API Key disimpan!")
                # Perbaikan: ganti experimental_rerun dengan rerun
                st.rerun()
        return None
    return st.session_state.fmp_api_key

# Fungsi untuk mendapatkan data real-time
def get_realtime_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="5m")
        if hist.empty:
            return None, None, None, None
        
        last_price = hist['Close'].iloc[-1]
        prev_close = stock.info.get('previousClose', last_price)
        change = last_price - prev_close
        change_percent = (change / prev_close) * 100
        
        return last_price, change, change_percent, hist
    except Exception as e:
        st.error(f"Error fetching real-time data: {str(e)}")
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

# Fungsi analisis DCA dengan data real-time
def dca_analysis(df):
    if df.empty:
        return
    
    st.subheader("📊 Analisis Dollar Cost Averaging (DCA)")
    
    # Hitung nilai investasi awal
    df['Total Investment'] = df['Lot Balance'] * df['Avg Price']
    total_investment = df['Total Investment'].sum()
    
    # Dapatkan harga real-time dan hitung nilai saat ini
    current_values = []
    current_prices = []
    changes = []
    
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        last_price, change, change_percent, _ = get_realtime_data(ticker)
        
        if last_price is not None:
            current_value = row['Lot Balance'] * last_price
            current_values.append(current_value)
            current_prices.append(last_price)
            changes.append(current_value - row['Total Investment'])
        else:
            current_values.append(0)
            current_prices.append(0)
            changes.append(0)
    
    df['Current Price'] = current_prices
    df['Current Value'] = current_values
    df['Profit/Loss'] = changes
    df['Profit/Loss %'] = (df['Current Value'] / df['Total Investment'] - 1) * 100
    
    total_current_value = df['Current Value'].sum()
    total_profit = total_current_value - total_investment
    total_profit_percent = (total_current_value / total_investment - 1) * 100
    
    # Tampilkan metrik utama
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
    df_display = df[['Ticker', 'Lot Balance', 'Avg Price', 'Current Price', 
                    'Total Investment', 'Current Value', 'Profit/Loss', 'Profit/Loss %']]
    
    st.dataframe(df_display.style.format({
        'Avg Price': 'Rp {:,.0f}',
        'Current Price': 'Rp {:,.0f}',
        'Total Investment': 'Rp {:,.0f}',
        'Current Value': 'Rp {:,.0f}',
        'Profit/Loss': 'Rp {:,.0f}',
        'Profit/Loss %': '{:+.2f}%'
    }), use_container_width=True)

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

# Sidebar menu
st.sidebar.title("📋 Menu Analisis")
st.sidebar.header("Konfigurasi API")
api_key = get_fmp_api_key()

st.sidebar.header("Portfolio")
uploaded_file = st.sidebar.file_uploader("Upload Portfolio", type=["csv", "xlsx"])

portfolio_df = pd.DataFrame()
if uploaded_file:
    portfolio_df = process_uploaded_file(uploaded_file)
    if not portfolio_df.empty:
        st.sidebar.success("File portfolio berhasil diupload!")
        st.sidebar.dataframe(portfolio_df[['Stock', 'Ticker']])

st.sidebar.header("Analisis")
menu_options = [
    "Dashboard Portfolio",
    "Analisis DCA",
    "Prediksi Harga Saham",
    "Valuasi Saham",
    "Tracking Modal"
]
selected_menu = st.sidebar.selectbox("Pilih Fitur:", menu_options)

# Main content area
st.title("📈 Stock Analysis Toolkit Pro")

if selected_menu == "Dashboard Portfolio":
    if not portfolio_df.empty:
        dca_analysis(portfolio_df)
    else:
        st.info("Silakan upload file portfolio untuk melihat dashboard")

elif selected_menu == "Analisis DCA":
    if not portfolio_df.empty:
        dca_analysis(portfolio_df)
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
        # Hapus suffix .JK untuk FMP
        clean_ticker = selected_ticker.replace('.JK', '')
        stock_valuation(clean_ticker, api_key)
    elif not api_key:
        st.warning("Silakan masukkan API Key FMP di sidebar")
    else:
        st.warning("Silakan upload file portfolio terlebih dahulu")

elif selected_menu == "Tracking Modal":
    capital_tracking()
