import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
from io import BytesIO
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Stock Analysis Toolkit Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk mendapatkan API key
def get_fmp_api_key():
    if 'fmp_api_key' not in st.session_state:
        with st.sidebar:
            st.subheader("FinancialModelingPrep API")
            api_key = st.text_input("Masukkan API Key FMP", type="password")
            if st.button("Simpan API Key"):
                st.session_state.fmp_api_key = api_key
                st.success("API Key disimpan!")
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
        last_price, _, _, _ = get_realtime_data(ticker)
        current_prices.append(last_price if last_price is not None else row['Avg Price'])
    
    df['Current Price'] = current_prices
    df['Current Value'] = df[lot_balance_col] * df['Current Price']
    df['Profit/Loss'] = df['Current Value'] - (df[lot_balance_col] * df['Avg Price'])
    df['Profit/Loss %'] = (df['Current Value'] / (df[lot_balance_col] * df['Avg Price']) - 1) * 100
    
    return df

# Fungsi analisis DCA dengan data real-time
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

# Fungsi untuk menghitung indikator teknikal
def calculate_technical_indicators(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        
        if hist.empty:
            return None, None, None
        
        # Hitung RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        
        # Hitung MACD
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Hitung MA Crossover
        ma50 = hist['Close'].rolling(window=50).mean()
        ma200 = hist['Close'].rolling(window=200).mean()
        
        # Cek crossover
        if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
            crossover = "Golden Cross"
        elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
            crossover = "Death Cross"
        else:
            crossover = "No Crossover"
        
        return last_rsi, macd.iloc[-1], crossover
    
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None, None, None

# Fungsi screening saham
def stock_screener(api_key):
    st.subheader("🔍 Stock Screener")
    
    # Inisialisasi session state untuk watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    # Form kriteria screening
    with st.form("screener_form"):
        col1, col2, col3 = st.columns(3)
        
        # Kriteria fundamental
        with col1:
            st.subheader("Fundamental")
            per_min = st.number_input("PER Min", value=0)
            per_max = st.number_input("PER Max", value=50)
            pbv_min = st.number_input("PBV Min", value=0)
            pbv_max = st.number_input("PBV Max", value=5)
            roe_min = st.number_input("ROE Min (%)", value=0)
            der_max = st.number_input("DER Max", value=5)
            growth_min = st.number_input("Revenue Growth Min (%)", value=0)
        
        # Kriteria teknikal
        with col2:
            st.subheader("Teknikal")
            rsi_min = st.number_input("RSI Min", value=30)
            rsi_max = st.number_input("RSI Max", value=70)
            macd_min = st.number_input("MACD Min", value=-1)
            macd_max = st.number_input("MACD Max", value=1)
            ma_crossover = st.selectbox("MA Crossover", 
                                       ["Any", "Golden Cross", "Death Cross", "No Crossover"])
        
        # Daftar saham yang akan di-scan
        with col3:
            st.subheader("Saham")
            stock_list = st.text_area("Masukkan kode saham (pisahkan dengan koma)", 
                                     "BBCA.JK, BBRI.JK, TLKM.JK, UNVR.JK")
            stock_list = [s.strip() for s in stock_list.split(',') if s.strip()]
        
        submit = st.form_submit_button("Screening Saham")
    
    if submit:
        results = []
        progress_bar = st.progress(0)
        total_stocks = len(stock_list)
        
        for i, ticker in enumerate(stock_list):
            progress_bar.progress((i+1)/total_stocks)
            
            # Dapatkan data fundamental
            fmp_data = None
            if api_key:
                clean_ticker = ticker.replace('.JK', '')
                fmp_data = get_fmp_data(clean_ticker, api_key)
            
            # Dapatkan data teknikal
            rsi, macd, crossover = calculate_technical_indicators(ticker)
            
            # Evaluasi kriteria
            fundamental_pass = True
            technical_pass = True
            
            if fmp_data:
                ratios = fmp_data['ratios']
                profile = fmp_data['profile']
                growth = fmp_data['growth']
                
                per = ratios.get('priceEarningsRatio', 0)
                pbv = ratios.get('priceToBookRatio', 0)
                roe = ratios.get('returnOnEquity', 0) * 100
                der = ratios.get('debtEquityRatio', 0)
                revenue_growth = growth.get('growthRevenue', 0) * 100 if growth else 0
                
                # Cek kriteria fundamental
                if per_min > per or per > per_max:
                    fundamental_pass = False
                if pbv_min > pbv or pbv > pbv_max:
                    fundamental_pass = False
                if roe < roe_min:
                    fundamental_pass = False
                if der > der_max:
                    fundamental_pass = False
                if revenue_growth < growth_min:
                    fundamental_pass = False
            
            # Cek kriteria teknikal
            if rsi and (rsi < rsi_min or rsi > rsi_max):
                technical_pass = False
            if macd and (macd < macd_min or macd > macd_max):
                technical_pass = False
            if ma_crossover != "Any" and crossover != ma_crossover:
                technical_pass = False
            
            # Tambahkan ke hasil jika memenuhi kriteria
            if fundamental_pass and technical_pass:
                results.append({
                    'Ticker': ticker,
                    'PER': per if fmp_data else 'N/A',
                    'PBV': pbv if fmp_data else 'N/A',
                    'ROE': roe if fmp_data else 'N/A',
                    'DER': der if fmp_data else 'N/A',
                    'Growth': revenue_growth if fmp_data else 'N/A',
                    'RSI': rsi if rsi else 'N/A',
                    'MACD': macd if macd else 'N/A',
                    'MA Crossover': crossover if crossover else 'N/A'
                })
        
        progress_bar.empty()
        
        if results:
            # Tampilkan hasil
            results_df = pd.DataFrame(results)
            
            # Format kolom
            results_display = results_df.style.format({
                'PER': '{:.2f}',
                'PBV': '{:.2f}',
                'ROE': '{:.2f}%',
                'DER': '{:.2f}',
                'Growth': '{:.2f}%',
                'RSI': '{:.2f}',
                'MACD': '{:.4f}'
            })
            
            st.dataframe(results_display, use_container_width=True)
            
            # Tombol untuk menambahkan semua ke watchlist
            if st.button("Tambahkan Semua ke Watchlist"):
                for item in results:
                    if item['Ticker'] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(item['Ticker'])
                st.success(f"{len(results)} saham ditambahkan ke watchlist!")
        else:
            st.warning("Tidak ada saham yang memenuhi kriteria screening")
    
    return stock_list

# Fungsi watchlist
def watchlist_manager():
    st.subheader("⭐ Watchlist Saham")
    
    # Inisialisasi session state untuk watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    # Form tambah saham ke watchlist
    with st.form("add_to_watchlist"):
        new_ticker = st.text_input("Tambah Saham (contoh: BBCA.JK)")
        if st.form_submit_button("Tambahkan ke Watchlist"):
            if new_ticker and new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)
                st.success(f"{new_ticker} ditambahkan ke watchlist!")
    
    # Tampilkan daftar watchlist
    if st.session_state.watchlist:
        # Dapatkan data real-time
        watchlist_data = []
        progress_bar = st.progress(0)
        total_stocks = len(st.session_state.watchlist)
        
        for i, ticker in enumerate(st.session_state.watchlist):
            progress_bar.progress((i+1)/total_stocks)
            price, change, change_percent, _ = get_realtime_data(ticker)
            
            if price is not None:
                watchlist_data.append({
                    'Ticker': ticker,
                    'Harga': price,
                    'Perubahan': change,
                    'Perubahan %': change_percent
                })
        
        progress_bar.empty()
        
        if watchlist_data:
            # Tampilkan tabel
            watchlist_df = pd.DataFrame(watchlist_data)
            watchlist_display = watchlist_df.style.format({
                'Harga': 'Rp {:,.0f}',
                'Perubahan': 'Rp {:,.0f}',
                'Perubahan %': '{:+.2f}%'
            }).apply(lambda x: ['background-color: lightgreen' if x['Perubahan'] > 0 
                              else 'background-color: lightcoral' for i in x], axis=1)
            
            st.dataframe(watchlist_display, use_container_width=True)
            
            # Grafik pergerakan harga
            st.subheader("Pergerakan Harga")
            fig = go.Figure()
            
            for ticker in st.session_state.watchlist:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1mo")
                if not hist.empty:
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        name=ticker
                    ))
            
            fig.update_layout(
                title='Pergerakan Harga 1 Bulan Terakhir',
                yaxis_title='Harga (Rp)',
                xaxis_title='Tanggal'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tombol hapus
            st.subheader("Kelola Watchlist")
            delete_ticker = st.selectbox("Pilih saham untuk dihapus", st.session_state.watchlist)
            if st.button(f"Hapus {delete_ticker} dari Watchlist"):
                st.session_state.watchlist.remove(delete_ticker)
                st.success(f"{delete_ticker} dihapus dari watchlist!")
        else:
            st.warning("Tidak ada data real-time untuk saham di watchlist")
    else:
        st.info("Watchlist kosong. Tambahkan saham untuk mulai memantau")

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
        
        # Inisialisasi watchlist dengan saham dari portfolio
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = portfolio_df['Ticker'].tolist()

st.sidebar.header("Analisis")
menu_options = [
    "Dashboard Portfolio",
    "Analisis DCA",
    "Prediksi Harga Saham",
    "Valuasi Saham",
    "Tracking Modal",
    "Rekomendasi Pembelian",
    "Screening Saham",  # Menu baru
    "Watchlist"         # Menu baru
]
selected_menu = st.sidebar.selectbox("Pilih Fitur:", menu_options)

# Main content area
st.title("📈 Stock Analysis Toolkit Pro")

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
        # Hapus suffix .JK untuk FMP
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

elif selected_menu == "Screening Saham":
    if api_key:
        stock_screener(api_key)
    else:
        st.warning("Silakan masukkan API Key FMP di sidebar")

elif selected_menu == "Watchlist":
    watchlist_manager()
