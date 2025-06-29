import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

# Konfigurasi awal
st.set_page_config(layout="wide", page_title="Analisis Portofolio Saham")

# Daftar saham indeks Kompas100 dan LQ45
KOMPAS100 = [
    'BBCA', 'BBRI', 'TLKM', 'BMRI', 'ASII', 'UNVR', 'PGAS', 'ADRO', 'CPIN', 'ANTM',
    'INDF', 'AKRA', 'KLBF', 'BBNI', 'UNTR', 'ICBP', 'SMGR', 'GGRM', 'PTBA', 'HMSP',
    'ITMG', 'MNCN', 'MEDC', 'INCO', 'LPKR', 'JPFA', 'LSIP', 'SIDO', 'SRIL', 'TPIA',
    'WIKA', 'WTON', 'AKPI', 'BUKA', 'EXCL', 'MDKA', 'MYRX', 'PTPP', 'SCMA', 'SMRA',
    'TINS', 'TOWR', 'TURI', 'ULTJ', 'UNIQ', 'WEGE', 'WSBP', 'WSKT', 'YELO'
]

LQ45 = [
    'ACES', 'ADRO', 'AKRA', 'AMRT', 'ANTM', 'ASII', 'BBCA', 'BBNI', 'BBRI', 'BBTN',
    'BMRI', 'BRPT', 'BUKA', 'CPIN', 'EMTK', 'ERAA', 'EXCL', 'GGRM', 'GOTO', 'HRUM',
    'ICBP', 'INCO', 'INDF', 'INKP', 'INTP', 'ITMG', 'JPFA', 'KLBF', 'MDKA', 'MEDC',
    'MIKA', 'MNCN', 'PGAS', 'PTBA', 'PTPP', 'SMGR', 'SMMA', 'TBIG', 'TINS', 'TKIM',
    'TLKM', 'TOWR', 'TPIA', 'UNTR', 'UNVR', 'WIKA', 'WSKT', 'WTON'
]

# Fungsi untuk memuat data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Bersihkan data
        df = df.dropna(subset=['Stock'])
        
        # Normalisasi ticker: pastikan format .JK dan huruf besar
        df['Ticker'] = df['Ticker'].str.upper().str.replace('.JK', '', regex=False) + '.JK'
        
        # Perbaikan konversi harga: hapus semua karakter non-digit
        df['Avg Price'] = (
            df['Avg Price']
            .astype(str)  # Pastikan berupa string
            .str.replace(r'[^\d]', '', regex=True)  # Hapus semua non-digit
            .replace('', np.nan)  # Ganti string kosong dengan NaN
            .dropna()  # Hapus baris dengan nilai NaN
            .astype(float)  # Konversi ke float
        )
        
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
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Fungsi untuk menghitung level support
def calculate_support_levels(data):
    support_levels = {}
    
    if not data.empty:
        closes = data['Close']
        
        # Menghitung moving averages dengan pandas
        if len(closes) >= 50:
            support_levels['MA50'] = closes.rolling(window=50).mean().iloc[-1]
        if len(closes) >= 100:
            support_levels['MA100'] = closes.rolling(window=100).mean().iloc[-1]
        if len(closes) >= 200:
            support_levels['MA200'] = closes.rolling(window=200).mean().iloc[-1]
        
        # Menghitung Fibonacci retracement levels
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        
        support_levels['Fib_23.6%'] = high - diff * 0.236
        support_levels['Fib_38.2%'] = high - diff * 0.382
        support_levels['Fib_50%'] = high - diff * 0.5
        support_levels['Fib_61.8%'] = high - diff * 0.618
        
        # Menghitung pivot points
        latest = data.iloc[-1]
        pivot = (latest['High'] + latest['Low'] + latest['Close']) / 3
        support_levels['Pivot_S1'] = (2 * pivot) - latest['High']
        support_levels['Pivot_S2'] = pivot - (latest['High'] - latest['Low'])
        support_levels['Pivot_S3'] = latest['Low'] - 2 * (latest['High'] - pivot)
        
        # Menambahkan harga terendah
        support_levels['1m_Low'] = data['Low'].tail(20).min()
        support_levels['3m_Low'] = data['Low'].tail(60).min()
        support_levels['52w_Low'] = data['Low'].min()
    
    # Hapus nilai NaN
    return {k: v for k, v in support_levels.items() if not pd.isna(v)}

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
            
            # Dapatkan data untuk support
            hist_data = get_stock_data(ticker, period="2y")
            support_levels = calculate_support_levels(hist_data) if not hist_data.empty else {}
            
            results.append({
                'Stock': row['Stock'],
                'Ticker': ticker,
                'Avg Price': avg_price,
                'Current Price': current_price,
                'DCA Analysis': dca_values,
                'Support Levels': support_levels,
                'Performance': (current_price - avg_price) / avg_price * 100
            })
    
    return pd.DataFrame(results)

# Fungsi untuk mendapatkan harga terkini
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else np.nan
    except Exception as e:
        print(f"Error getting price for {ticker}: {str(e)}")
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
    except Exception as e:
        print(f"Error in DCA simulation for {ticker}: {str(e)}")
        return np.nan

# Fungsi untuk mendapatkan sentimen berita
def get_news_sentiment(ticker):
    try:
        # Ganti dengan implementasi API berita sebenarnya
        # Contoh placeholder
        time.sleep(0.1)  # Delay untuk simulasi (diperpendek)
        
        # Simulasi sentimen acak
        sentiments = ['Positif', 'Netral', 'Negatif']
        return np.random.choice(sentiments, p=[0.3, 0.5, 0.2])
    except:
        return "Tidak tersedia"

# Fungsi untuk mendapatkan data saham rekomendasi
def get_recommended_stocks(index_name):
    # Tentukan indeks yang dipilih
    stocks = KOMPAS100 if index_name == "Kompas100" else LQ45
    
    # Dapatkan data untuk setiap saham
    data = []
    for stock in stocks:
        try:
            ticker = f"{stock}.JK"
            price = get_current_price(ticker)
            if not np.isnan(price):
                # Simulasi data dividen
                dividend_yield = np.random.uniform(1.5, 6.0)
                
                # Simulasi support level
                support_level = price * (1 - np.random.uniform(0.05, 0.15))
                
                data.append({
                    'Saham': stock,
                    'Ticker': ticker,
                    'Harga (Rp)': price,
                    'Dividen Yield (%)': dividend_yield,
                    'Support Terdekat (Rp)': support_level
                })
        except:
            continue
    
    # Buat DataFrame dan urutkan berdasarkan dividen yield
    df = pd.DataFrame(data)
    df = df.sort_values('Dividen Yield (%)', ascending=False)
    return df.head(10)  # Ambil 10 teratas

# Fungsi untuk mengalokasikan modal ke saham rekomendasi
def allocate_capital(capital, recommended_stocks):
    # Hitung total dividen yield untuk pembobotan
    total_yield = recommended_stocks['Dividen Yield (%)'].sum()
    
    # Alokasikan modal berdasarkan bobot dividen yield
    recommended_stocks['Bobot'] = recommended_stocks['Dividen Yield (%)'] / total_yield
    recommended_stocks['Alokasi Modal'] = recommended_stocks['Bobot'] * capital
    
    # Hitung jumlah lot (1 lot = 100 lembar)
    recommended_stocks['Jumlah Lot'] = (recommended_stocks['Alokasi Modal'] / 
                                       (recommended_stocks['Harga (Rp)'] * 100)).apply(np.floor)
    
    # Hitung nilai investasi aktual
    recommended_stocks['Nilai Investasi'] = recommended_stocks['Jumlah Lot'] * 100 * recommended_stocks['Harga (Rp)']
    
    # Hitung sisa uang
    recommended_stocks['Sisa Uang'] = recommended_stocks['Alokasi Modal'] - recommended_stocks['Nilai Investasi']
    
    # Hitung yield estimasi
    recommended_stocks['Estimasi Dividen Tahunan'] = (recommended_stocks['Nilai Investasi'] * 
                                                    recommended_stocks['Dividen Yield (%)'] / 100)
    
    return recommended_stocks

# Fungsi analisis rekomendasi portofolio (diperbaiki)
def analyze_portfolio(df):
    recommendations = []
    for _, row in df.iterrows():
        ticker = row['Ticker']
        avg_price = row['Avg Price']
        current_price = get_current_price(ticker)
        
        if not np.isnan(current_price):
            performance = (current_price - avg_price) / avg_price * 100
            
            # Dapatkan level support
            hist_data = get_stock_data(ticker, period="1y")
            support_levels = calculate_support_levels(hist_data) if not hist_data.empty else {}
            
            # Tambahkan pertimbangan support dalam rekomendasi
            support_reason = ""
            support_level = None
            
            if support_levels:
                # Urutkan level support dari yang terdekat dengan harga saat ini
                sorted_supports = sorted(
                   [(k, v) for k, v in row['Support Levels'].items()], 
                   key=lambda x: abs(x[1] - current_price))
                
                # Ambil 3 level support terdekat
                closest_supports = sorted_supports[:3]
                support_level = closest_supports[0][1] if closest_supports else None
                
                # Hitung jarak ke support terdekat
                if support_level:
                    distance = (current_price - support_level) / current_price * 100
                    support_reason = f"Support terdekat: {closest_supports[0][0]} (Rp {support_level:,.0f}, {distance:+.2f}%)"
            
            if performance > 25:
                rec = 'JUAL'
                reason = f'Kenaikan signifikan ({performance:.2f}%)'
                color = 'red'
            elif performance < -15:
                rec = 'TAMBAH'
                reason = f'Potensi averaging down ({performance:.2f}%)'
                color = 'green'
            elif performance > 10:
                rec = 'HOLD'
                reason = f'Kinerja baik ({performance:.2f}%)'
                color = 'yellow'
            else:
                rec = 'HOLD'
                reason = 'Performa wajar'
                color = 'yellow'
            
            # Jika dekat dengan level support, pertimbangkan untuk tambah
            if support_level and (current_price - support_level) / current_price < 0.05:
                if rec == 'HOLD':
                    rec = 'TAMBAH'
                    reason += f" | Dekat support ({support_reason})"
                    color = 'green'
            
            recommendations.append({
                'Saham': row['Stock'],
                'Ticker': ticker,
                'Rekomendasi': rec,
                'Alasan': reason,
                'Warna': color,
                'Support Level': support_reason,
                'Harga Avg (Rp)': avg_price,
                'Harga Sekarang (Rp)': current_price,
                'Performa (%)': performance
            })
    
    return pd.DataFrame(recommendations)
    
# Fungsi untuk menampilkan grafik dengan support levels
def plot_with_support(data, support_levels, ticker):
    if data.empty or not support_levels:
        return None
    
    fig = go.Figure()
    
    # Tambahkan garis harga
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'],
        mode='lines',
        name='Harga Penutupan',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Tambahkan garis support
    colors = px.colors.qualitative.Plotly
    for i, (level_name, level_value) in enumerate(support_levels.items()):
        fig.add_hline(
            y=level_value,
            line=dict(color=colors[i % len(colors)], dash='dash'),
            annotation_text=f"{level_name}: {level_value:,.0f}",
            annotation_position="bottom right"
        )
    
    # Konfigurasi layout
    fig.update_layout(
        title=f'Performa Saham {ticker} dengan Level Support',
        xaxis_title='Tanggal',
        yaxis_title='Harga (Rp)',
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    return fig

# FUNGSI BARU: Menentukan rekomendasi pembelian dengan indikator warna
def get_purchase_recommendation(current_price, support_levels, avg_price):
    """
    Menentukan rekomendasi pembelian berdasarkan analisis teknikal:
    - Hijau (wajib beli): Harga di bawah support terdekat atau di bawah MA200
    - Kuning (beli sebagian): Harga di atas support tetapi di bawah rata-rata portofolio
    - Merah (amati): Harga di atas semua level support dan rata-rata portofolio
    """
    if not support_levels:
        return "Data tidak cukup", "gray"
    
    # Ambil level support terdekat
    min_support = min(support_levels.values())
    
    # Analisis kondisi
    if current_price < min_support:
        return "Wajib Beli (di bawah support)", "green"
    elif current_price < min_support * 1.05:  # Dalam 5% di atas support terendah
        return "Beli Sebagian (dekat support)", "orange"
    elif current_price < avg_price:
        return "Beli Sebagian (di bawah rata-rata)", "orange"
    else:
        return "Amati Pasar (di atas support)", "red"

# Fungsi untuk menghitung jumlah lot yang bisa dibeli (DIPERBARUI)
def calculate_lot_purchase(df, capital, dca_df):
    results = []
    total_cost = 0
    
    for _, row in df.iterrows():
        ticker = row['Ticker']
        current_price = get_current_price(ticker)
        
        if not np.isnan(current_price):
            # Hitung jumlah lot yang bisa dibeli (1 lot = 100 lembar)
            lot_size = 100
            price_per_lot = current_price * lot_size
            max_lots = capital // price_per_lot
            
            # Hitung biaya pembelian untuk semua lot yang bisa dibeli
            cost = max_lots * price_per_lot
            
            # Dapatkan data support dari dca_df
            support_levels = {}
            dca_row = dca_df[dca_df['Ticker'] == ticker]
            if not dca_row.empty:
                support_levels = dca_row.iloc[0]['Support Levels']
            
            # Dapatkan rekomendasi pembelian
            rec_text, rec_color = get_purchase_recommendation(
                current_price, 
                support_levels,
                row['Avg Price']
            )
            
            results.append({
                'Saham': row['Stock'],
                'Ticker': ticker,
                'Harga Sekarang (Rp)': current_price,
                'Harga per Lot (Rp)': price_per_lot,
                'Jumlah Lot yang Bisa Dibeli': max_lots,
                'Total Biaya (Rp)': cost,
                'Rekomendasi Pembelian': rec_text,
                'Warna': rec_color
            })
            
            total_cost += cost
    
    # Hitung sisa modal
    remaining_capital = capital - total_cost
    
    return pd.DataFrame(results), total_cost, remaining_capital

# FITUR BARU: Simulasi Pensiun / Tujuan Keuangan
def retirement_simulation(initial_capital, annual_return, annual_dividend, years, monthly_contribution=0):
    # Konversi persentase ke desimal
    annual_return_decimal = annual_return / 100.0
    annual_dividend_decimal = annual_dividend / 100.0
    
    # Data untuk grafik
    years_list = []
    capital_growth = []
    dividend_income = []
    total_value = []
    monthly_dividends = []
    
    # Inisialisasi nilai awal
    current_capital = initial_capital
    total_dividend = 0
    
    # Simulasi tahunan
    for year in range(1, years + 1):
        # Pertumbuhan modal
        capital_growth.append(current_capital)
        
        # Hitung dividen tahunan
        dividend = current_capital * annual_dividend_decimal
        dividend_income.append(dividend)
        total_dividend += dividend
        
        # Reinvestasi dividen
        current_capital += dividend
        
        # Pertumbuhan modal (capital gain)
        current_capital *= (1 + annual_return_decimal)
        
        # Tambahkan kontribusi bulanan
        current_capital += monthly_contribution * 12
        
        # Hitung pendapatan dividen per bulan
        monthly_dividend = dividend / 12
        
        years_list.append(year)
        total_value.append(current_capital)
        monthly_dividends.append(monthly_dividend)
    
    # Buat DataFrame hasil simulasi
    df = pd.DataFrame({
        'Tahun': years_list,
        'Pertumbuhan Modal (Rp)': capital_growth,
        'Pendapatan Dividen Tahunan (Rp)': dividend_income,
        'Total Nilai Portofolio (Rp)': total_value,
        'Pendapatan Dividen Bulanan (Rp)': monthly_dividends
    })
    
    return df, total_dividend

# Tampilan Streamlit
st.title("📈 Analisis Portofolio Saham & Simulasi Keuangan")

# Buat tab untuk navigasi
tab1, tab2 = st.tabs(["Analisis Portofolio", "Simulasi Pensiun"])

with tab1:
    # Input modal dan indeks di sidebar
    st.sidebar.header("Alokasi Modal")
    capital = st.sidebar.number_input("Modal yang Tersedia (Rp)", min_value=1000000, value=50000000, step=1000000)
    index_selection = st.sidebar.selectbox("Pilih Indeks Saham", ["Kompas100", "LQ45"])
    calculate_allocation = st.sidebar.button("Hitung Alokasi Modal")
    calculate_lot = st.sidebar.button("Hitung Jumlah Lot yang Bisa Dibeli")

    # Upload file
    uploaded_file = st.file_uploader("Unggah file portofolio saham (CSV)", type="csv")
    df = pd.DataFrame()
    dca_df = pd.DataFrame()  # Simpan hasil analisis DCA

    if uploaded_file:
        # Gunakan fungsi load_data yang sudah didefinisikan
        df = load_data(uploaded_file)
        
        if not df.empty:
            st.success("Data berhasil dimuat!")
            st.subheader("Portofolio Saat Ini")
            st.dataframe(df.style.format({'Avg Price': 'Rp {:,.0f}'}), height=300)
            
            # Analisis DCA dan Support
            st.subheader("🔍 Analisis Dollar Cost Averaging (DCA) & Support Level")
            dca_df = dca_analysis(df)
            
            if not dca_df.empty:
                # Tampilkan hasil DCA dan Support
                for _, row in dca_df.iterrows():
                    with st.expander(f"{row['Stock']} ({row['Ticker']})", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Harga Rata-rata:** Rp {row['Avg Price']:,.0f}")
                            st.write(f"**Harga Sekarang:** Rp {row['Current Price']:,.0f}")
                            st.write(f"**Performa:** {row['Performance']:.2f}%")
                            
                            # Tampilkan analisis DCA
                            st.subheader("Simulasi DCA")
                            for dca in row['DCA Analysis']:
                                st.write(f"- Rata-rata {dca['period']}: Rp {dca['simulated_price']:,.0f}")
                        
                        with col2:
                            # Tampilkan level support
                            if row['Support Levels']:
                                st.subheader("Level Support Penting")
                                support_df = pd.DataFrame(
                                    list(row['Support Levels'].items()), 
                                    columns=['Level', 'Harga (Rp)']
                                ).sort_values('Harga (Rp)', ascending=False)
                                
                                st.dataframe(support_df.style.format({'Harga (Rp)': 'Rp {:,.0f}'}))
                            else:
                                st.warning("Data support tidak tersedia")
                        
                        # Grafik dengan level support
                        if row['Support Levels']:
                            st.subheader("Grafik Harga dengan Level Support")
                            hist_data = get_stock_data(row['Ticker'], period="1y")
                            fig = plot_with_support(hist_data, row['Support Levels'], row['Ticker'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
            
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
                # Format warna untuk rekomendasi
                def color_recommendation(row):
                    colors = {
                        'green': 'lightgreen',
                        'yellow': 'lightyellow',
                        'red': 'salmon'
                    }
                    return [f'background-color: {colors.get(row["Warna"], "white")}'] * len(row)
                
                st.dataframe(
                    rec_df.style.apply(color_recommendation, axis=1)
                    .format({
                        'Harga Avg (Rp)': 'Rp {:,.0f}',
                        'Harga Sekarang (Rp)': 'Rp {:,.0f}',
                        'Performa (%)': '{:.2f}%'
                    }),
                    height=400
                )
            
            # FITUR BARU: Hitung jumlah lot yang bisa dibeli dengan rekomendasi warna
            if calculate_lot and capital > 0:
                st.markdown("---")
                st.subheader("🛒 Perhitungan Jumlah Lot yang Bisa Dibeli")
                st.write(f"Modal yang tersedia: Rp {capital:,.0f}")
                
                if not dca_df.empty:
                    lot_df, total_cost, remaining_capital = calculate_lot_purchase(df, capital, dca_df)
                    
                    if not lot_df.empty:
                        # Fungsi untuk styling rekomendasi
                        def color_purchase_recommendation(row):
                            colors = {
                                'green': 'lightgreen',
                                'orange': 'moccasin',
                                'red': 'salmon',
                                'gray': 'lightgray'
                            }
                            return [f'background-color: {colors.get(row["Warna"], "white")}'] * len(row)
                        
                        # Tampilkan hasil perhitungan lot
                        st.write("### Perhitungan Pembelian Saham")
                        st.dataframe(
                            lot_df.style.apply(color_purchase_recommendation, axis=1)
                            .format({
                                'Harga Sekarang (Rp)': 'Rp {:,.0f}',
                                'Harga per Lot (Rp)': 'Rp {:,.0f}',
                                'Total Biaya (Rp)': 'Rp {:,.0f}'
                            }),
                            height=500
                        )
                        
                        # Tampilkan ringkasan
                        col1, col2 = st.columns(2)
                        col1.metric("Total Biaya Pembelian", f"Rp {total_cost:,.0f}")
                        col2.metric("Sisa Modal", f"Rp {remaining_capital:,.0f}")
                        
                        # Grafik alokasi pembelian
                        fig = px.bar(
                            lot_df.sort_values('Total Biaya (Rp)', ascending=False),
                            x='Saham',
                            y='Jumlah Lot yang Bisa Dibeli',
                            title='Jumlah Lot yang Bisa Dibeli per Saham',
                            color='Rekomendasi Pembelian',
                            color_discrete_map={
                                'Wajib Beli (di bawah support)': 'green',
                                'Beli Sebagian (dekat support)': 'orange',
                                'Beli Sebagian (di bawah rata-rata)': 'orange',
                                'Amati Pasar (di atas support)': 'red',
                                'Data tidak cukup': 'gray'
                            },
                            text='Jumlah Lot yang Bisa Dibeli'
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Tidak ada data untuk ditampilkan")
                else:
                    st.warning("Silakan lakukan analisis DCA terlebih dahulu")
            
            # Ringkasan Portofolio
            st.subheader("📊 Ringkasan Portofolio")
            if not rec_df.empty and 'Lot Balance' in df.columns:
                df['Total Investasi'] = df['Lot Balance'] * df['Avg Price']
                
                # Hitung nilai sekarang
                current_prices = rec_df.set_index('Ticker')['Harga Sekarang (Rp)']
                df['Current Price'] = df['Ticker'].map(current_prices)
                df['Nilai Sekarang'] = df['Lot Balance'] * df['Current Price']
                
                total_investment = df['Total Investasi'].sum()
                current_value = df['Nilai Sekarang'].sum()
                performance = (current_value - total_investment) / total_investment * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Investasi", f"Rp {total_investment:,.0f}")
                col2.metric("Nilai Sekarang", f"Rp {current_value:,.0f}", f"{performance:.2f}%")
                col3.metric("Profit/Rugi", f"Rp {current_value - total_investment:,.0f}", 
                           delta_color="inverse" if performance < 0 else "normal")
                
                # Grafik alokasi
                allocation = df.copy()
                allocation['Nilai'] = allocation['Nilai Sekarang']
                fig = px.pie(
                    allocation, 
                    names='Stock', 
                    values='Nilai',
                    title='Alokasi Portofolio Berdasarkan Nilai Pasar',
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Analisis support seluruh portofolio
                st.subheader("🛡️ Level Support Portofolio")
                support_data = []
                for _, row in dca_df.iterrows():
                    if row['Support Levels']:
                        # Ambil 3 support terdekat
                        sorted_supports = sorted(
                            [(k, v) for k, v in row['Support Levels'].items()], 
                            key=lambda x: abs(x[1] - row['Current Price'])
                        for i, (level, value) in enumerate(sorted_supports[:3]):
                            distance = (row['Current Price'] - value) / row['Current Price'] * 100
                            support_data.append({
                                'Saham': row['Stock'],
                                'Level Support': level,
                                'Harga Support': value,
                                'Harga Sekarang': row['Current Price'],
                                'Jarak (%)': distance,
                                'Kekuatan': 3 - i  # Semakin dekat semakin tinggi kekuatannya
                            })
                
                if support_data:
                    support_df = pd.DataFrame(support_data)
                    fig = px.bar(
                        support_df,
                        x='Saham',
                        y='Jarak (%)',
                        color='Level Support',
                        title='Jarak Harga ke Level Support Terdekat',
                        text='Harga Support',
                        hover_data=['Harga Support', 'Harga Sekarang']
                    )
                    fig.update_traces(texttemplate='Rp %{text:,.0f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Data support tidak tersedia untuk portofolio ini")
            else:
                st.warning("Tidak ada data rekomendasi untuk ditampilkan atau kolom 'Lot Balance' tidak ditemukan")

    else:
        st.info("Silakan unggah file portofolio saham dalam format CSV")

    # Rekomendasi Alokasi Modal
    if calculate_allocation and capital > 0:
        st.markdown("---")
        st.subheader(f"💼 Rekomendasi Alokasi Modal ({index_selection})")
        st.write(f"Modal yang tersedia: Rp {capital:,.0f}")
        
        # Dapatkan saham rekomendasi
        recommended_stocks = get_recommended_stocks(index_selection)
        
        if not recommended_stocks.empty:
            # Alokasikan modal
            allocation = allocate_capital(capital, recommended_stocks)
            
            # Tampilkan hasil alokasi
            st.write("### Alokasi Saham dan Jumlah Lot yang Dibeli")
            allocation_display = allocation[[
                'Saham', 'Harga (Rp)', 'Dividen Yield (%)', 'Support Terdekat (Rp)', 
                'Jumlah Lot', 'Nilai Investasi', 'Estimasi Dividen Tahunan'
            ]]
            
            # Format tampilan
            formatted_df = allocation_display.copy()
            formatted_df.columns = [
                'Saham', 'Harga (Rp)', 'Dividen Yield (%)', 'Support Terdekat (Rp)',
                'Jumlah Lot', 'Nilai Investasi (Rp)', 'Estimasi Dividen Tahunan (Rp)'
            ]
            
            st.dataframe(
                formatted_df.style.format({
                    'Harga (Rp)': 'Rp {:,.0f}',
                    'Support Terdekat (Rp)': 'Rp {:,.0f}',
                    'Dividen Yield (%)': '{:.2f}%',
                    'Nilai Investasi (Rp)': 'Rp {:,.0f}',
                    'Estimasi Dividen Tahunan (Rp)': 'Rp {:,.0f}'
                }),
                height=500
            )
            
            # Ringkasan alokasi
            total_investment = allocation['Nilai Investasi'].sum()
            total_dividen = allocation['Estimasi Dividen Tahunan'].sum()
            sisa_uang = capital - total_investment
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investasi", f"Rp {total_investment:,.0f}")
            col2.metric("Estimasi Dividen Tahunan", f"Rp {total_dividen:,.0f}")
            col3.metric("Sisa Uang", f"Rp {sisa_uang:,.0f}")
            
            # Grafik alokasi
            fig = px.pie(
                allocation,
                names='Saham',
                values='Nilai Investasi',
                title='Alokasi Modal per Saham',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Grafik estimasi dividen
            fig = px.bar(
                allocation.sort_values('Estimasi Dividen Tahunan', ascending=False),
                x='Saham',
                y='Estimasi Dividen Tahunan',
                title='Estimasi Dividen Tahunan per Saham',
                color='Dividen Yield (%)',
                text='Estimasi Dividen Tahunan'
            )
            fig.update_traces(texttemplate='Rp %{text:,.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak dapat menemukan data saham untuk indeks ini")
    elif calculate_allocation:
        st.warning("Masukkan jumlah modal yang valid")

with tab2:
    st.header("🚀 Simulasi Pensiun / Tujuan Keuangan")
    st.write("""
    Proyeksikan pertumbuhan portofolio Anda hingga masa pensiun berdasarkan:
    - Return tahunan yang diharapkan
    - Dividen tahunan
    - Kontribusi bulanan
    """)
    
    # Input parameter simulasi
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Modal Awal (Rp)", min_value=0, value=100000000, step=1000000)
        annual_return = st.slider("Return Tahunan yang Diharapkan (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
        years = st.slider("Jangka Waktu (Tahun)", min_value=1, max_value=50, value=20, step=1)
    with col2:
        annual_dividend = st.slider("Dividen Tahunan (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        monthly_contribution = st.number_input("Kontribusi Bulanan (Rp)", min_value=0, value=1000000, step=100000)
    
    run_simulation = st.button("Proyeksikan Pertumbuhan Portofolio")
    
    if run_simulation:
        st.markdown("---")
        st.subheader(f"Proyeksi Pertumbuhan Portofolio ({years} Tahun)")
        
        # Jalankan simulasi
        simulation_df, total_dividend = retirement_simulation(
            initial_capital, 
            annual_return, 
            annual_dividend, 
            years,
            monthly_contribution
        )
        
        # Tampilkan ringkasan
        final_value = simulation_df['Total Nilai Portofolio (Rp)'].iloc[-1]
        monthly_dividend = simulation_df['Pendapatan Dividen Bulanan (Rp)'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Nilai Akhir Portofolio", f"Rp {final_value:,.0f}")
        col2.metric("Pendapatan Dividen Bulanan", f"Rp {monthly_dividend:,.0f}")
        col3.metric("Total Dividen Diterima", f"Rp {total_dividend:,.0f}")
        
        # Grafik pertumbuhan modal
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=simulation_df['Tahun'],
            y=simulation_df['Pertumbuhan Modal (Rp)'],
            name='Pertumbuhan Modal',
            line=dict(color='blue', width=3)
        ))
        fig1.add_trace(go.Scatter(
            x=simulation_df['Tahun'],
            y=simulation_df['Total Nilai Portofolio (Rp)'],
            name='Total Portofolio',
            line=dict(color='green', width=3, dash='dash')
        ))
        fig1.update_layout(
            title='Proyeksi Pertumbuhan Modal',
            xaxis_title='Tahun',
            yaxis_title='Nilai (Rp)',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Grafik pendapatan dividen
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=simulation_df['Tahun'],
            y=simulation_df['Pendapatan Dividen Tahunan (Rp)'],
            name='Dividen Tahunan',
            line=dict(color='orange', width=3)
        ))
        fig2.add_trace(go.Bar(
            x=simulation_df['Tahun'],
            y=simulation_df['Pendapatan Dividen Bulanan (Rp)'],
            name='Dividen Bulanan',
            marker_color='gold'
        ))
        fig2.update_layout(
            title='Proyeksi Pendapatan Dividen',
            xaxis_title='Tahun',
            yaxis_title='Dividen (Rp)',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Tampilkan tabel data
        with st.expander("Lihat Detail Proyeksi Tahunan", expanded=False):
            st.dataframe(
                simulation_df.style.format({
                    'Pertumbuhan Modal (Rp)': 'Rp {:,.0f}',
                    'Pendapatan Dividen Tahunan (Rp)': 'Rp {:,.0f}',
                    'Total Nilai Portofolio (Rp)': 'Rp {:,.0f}',
                    'Pendapatan Dividen Bulanan (Rp)': 'Rp {:,.0f}'
                })
            )

# Catatan kaki
st.markdown("---")
st.caption("© 2023 Tools Analisis Saham | Data harga saham bersumber dari Yahoo Finance | Support level dihitung menggunakan moving average, Fibonacci, dan pivot points")
