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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import logging

# Konfigurasi awal
st.set_page_config(layout="wide", page_title="Analisis Portofolio Saham")

# Setup logging untuk Prophet
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

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
            
            # Dapatkan level support secara langsung
            hist_data = get_stock_data(ticker, period="1y")
            support_levels = calculate_support_levels(hist_data) if not hist_data.empty else {}
            
            # Tambahkan pertimbangan support dalam rekomendasi
            support_reason = ""
            support_level = None
            
            if support_levels:
                # Urutkan level support dari yang terdekat dengan harga saat ini
                sorted_supports = sorted(
                    [(k, v) for k, v in support_levels.items()], 
                    key=lambda x: abs(x[1] - current_price))
                
                # Ambil 3 level support terdekat
                closest_supports = sorted_supports[:3]
                support_level = closest_supports[0][1] if closest_supports else None
                
                # Hitung jarak ke support terdekat
                if support_level:
                    distance = (current_price - support_level) / current_price * 100
                    support_reason = f"Support terdekat: {closest_supports[0][0]} (Rp {support_level:,.0f}, {distance:+.2f}%)"
            
            # ... (kode selanjutnya tetap sama) ...
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

# ========================================================
# FUNGSI BARU: PREDIKSI HARGA SAHAM
# ========================================================

# Fungsi untuk prediksi menggunakan LSTM
def predict_with_lstm(ticker, months=6):
    try:
        # Dapatkan data historis
        data = get_stock_data(ticker, period="2y")
        if data.empty or len(data) < 100:
            return None, None, "Data historis tidak cukup untuk prediksi"
        
        # Gunakan hanya kolom Close
        dataset = data[['Close']].values
        
        # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Buat dataset untuk training
        look_back = 60  # Gunakan 60 hari sebelumnya untuk prediksi
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Bangun model LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Latih model
        model.fit(X, y, epochs=25, batch_size=32, verbose=0)
        
        # Prediksi masa depan
        future_days = 30 * months  # Asumsi 30 hari per bulan
        predictions = []
        last_sequence = scaled_data[-look_back:]
        
        for _ in range(future_days):
            x_input = last_sequence.reshape(1, look_back, 1)
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred[0,0])
            last_sequence = np.append(last_sequence[1:], pred)
        
        # Invers transformasi
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Buat tanggal prediksi
        last_date = data.index[-1]
        pred_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]
        
        return predictions.flatten(), pred_dates, "Sukses"
    
    except Exception as e:
        print(f"Error in LSTM prediction: {str(e)}")
        return None, None, f"Error: {str(e)}"

# ========================================================
# FUNGSI BARU: VALUASI SAHAM
# ========================================================

def get_financial_data(ticker):
    """Mendapatkan data fundamental saham dari Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Data yang diperlukan
        current_price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
        book_value = info.get('bookValue', np.nan)
        eps = info.get('trailingEps', info.get('epsTrailingTwelveMonths', np.nan))
        pe_ratio = info.get('trailingPE', np.nan)
        pb_ratio = info.get('priceToBook', np.nan)
        dividend_yield = info.get('dividendYield', np.nan) * 100 if info.get('dividendYield') else np.nan
        beta = info.get('beta', np.nan)
        roe = info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan
        
        # Dapatkan data dividen
        dividends = stock.dividends
        if not dividends.empty:
            last_dividend = dividends.iloc[-1]
        else:
            last_dividend = np.nan
        
        return {
            'current_price': current_price,
            'book_value': book_value,
            'eps': eps,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'dividend_yield': dividend_yield,
            'last_dividend': last_dividend,
            'beta': beta,
            'roe': roe
        }
    except Exception as e:
        st.error(f"Error mendapatkan data fundamental untuk {ticker}: {str(e)}")
        return {}

def pbv_valuation(book_value, industry_pb, safety_margin=0.15):
    """Valuasi menggunakan Price to Book Value (PBV)"""
    if np.isnan(book_value) or np.isnan(industry_pb):
        return np.nan, "Data tidak lengkap"
    
    fair_value = book_value * industry_pb
    target_buy_price = fair_value * (1 - safety_margin)
    
    return fair_value, target_buy_price

def per_valuation(eps, industry_pe, growth_rate=0.1, safety_margin=0.15):
    """Valuasi menggunakan Price to Earnings Ratio (PER)"""
    if np.isnan(eps) or np.isnan(industry_pe):
        return np.nan, "Data tidak lengkap"
    
    # Hitung PER wajar dengan mempertimbangkan growth rate
    fair_pe = industry_pe * (1 + growth_rate)
    fair_value = eps * fair_pe
    target_buy_price = fair_value * (1 - safety_margin)
    
    return fair_value, target_buy_price

def dcf_valuation(eps, growth_rate_5y, terminal_growth_rate, discount_rate, safety_margin=0.15):
    """Valuasi menggunakan Discounted Cash Flow (DCF)"""
    if np.isnan(eps) or np.isnan(growth_rate_5y) or np.isnan(terminal_growth_rate) or np.isnan(discount_rate):
        return np.nan, "Data tidak lengkap"
    
    # Hitung arus kas bebas (FCFF) - disederhanakan dari EPS
    fcff = eps
    
    # Proyeksi 5 tahun dengan pertumbuhan tinggi
    cash_flows = []
    for year in range(1, 6):
        fcff *= (1 + growth_rate_5y)
        discounted_fcff = fcff / ((1 + discount_rate) ** year)
        cash_flows.append(discounted_fcff)
    
    # Terminal value
    terminal_value = (fcff * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** 5)
    
    # Total nilai perusahaan
    enterprise_value = sum(cash_flows) + discounted_terminal_value
    
    # Asumsikan nilai ekuitas sama dengan enterprise value
    fair_value = enterprise_value
    target_buy_price = fair_value * (1 - safety_margin)
    
    return fair_value, target_buy_price

def graham_valuation(eps, book_value_per_share, growth_rate_7y=0.15, bond_yield=0.07):
    """Valuasi menggunakan formula Benjamin Graham"""
    if np.isnan(eps) or np.isnan(book_value_per_share):
        return np.nan, "Data tidak lengkap"
    
    # Hitung PER wajar berdasarkan formula Graham
    fair_pe = (8.5 + 2 * growth_rate_7y) * (bond_yield / 0.04)
    fair_value = eps * fair_pe
    
    # Pastikan valuasi minimal 1.5x book value
    min_value = 1.5 * book_value_per_share
    if fair_value < min_value:
        fair_value = min_value
    
    # Margin of safety 25%
    target_buy_price = fair_value * 0.75
    
    return fair_value, target_buy_price

def display_valuation_results(ticker, valuation_data):
    """Menampilkan hasil valuasi dalam format tabel"""
    st.subheader(f"Hasil Valuasi untuk {ticker}")
    
    # Buat DataFrame untuk hasil valuasi
    methods = []
    fair_values = []
    buy_prices = []
    current_prices = []
    margins = []
    
    current_price = valuation_data['current_price']
    
    for method, (fair_value, buy_price) in valuation_data['results'].items():
        if not np.isnan(fair_value):
            methods.append(method)
            fair_values.append(fair_value)
            buy_prices.append(buy_price)
            current_prices.append(current_price)
            
            # Hitung margin of safety
            if current_price > 0:
                margin = ((fair_value - current_price) / current_price) * 100
            else:
                margin = np.nan
            margins.append(margin)
    
    df = pd.DataFrame({
        'Metode': methods,
        'Harga Wajar (Rp)': fair_values,
        'Harga Beli Target (Rp)': buy_prices,
        'Harga Sekarang (Rp)': current_prices,
        'Margin of Safety (%)': margins
    })
    
    # Tampilkan tabel
    st.dataframe(
        df.style.format({
            'Harga Wajar (Rp)': 'Rp {:,.0f}',
            'Harga Beli Target (Rp)': 'Rp {:,.0f}',
            'Harga Sekarang (Rp)': 'Rp {:,.0f}',
            'Margin of Safety (%)': '{:.2f}%'
        }),
        height=300
    )
    
    # Tampilkan rekomendasi
    avg_fair_value = np.nanmean(fair_values)
    if not np.isnan(avg_fair_value) and not np.isnan(current_price):
        if current_price < avg_fair_value * 0.8:
            st.success("**REKOMENDASI: BELI** - Harga saat ini di bawah nilai wajar dengan margin keamanan yang baik")
        elif current_price < avg_fair_value:
            st.warning("**REKOMENDASI: TAHAN** - Harga saat ini mendekati nilai wajar")
        else:
            st.error("**REKOMENDASI: JUAL** - Harga saat ini di atas nilai wajar")
        
        st.metric("Rata-rata Harga Wajar", f"Rp {avg_fair_value:,.0f}", 
                 f"{(avg_fair_value - current_price)/current_price*100:.2f}%")
    
    # Grafik perbandingan valuasi
    if len(methods) > 0:
        fig = go.Figure()
        
        # Tambahkan garis harga wajar
        fig.add_trace(go.Bar(
            x=methods,
            y=fair_values,
            name='Harga Wajar',
            marker_color='#2ca02c'
        ))
        
        # Tambahkan garis harga beli target
        fig.add_trace(go.Bar(
            x=methods,
            y=buy_prices,
            name='Harga Beli Target',
            marker_color='#ff7f0e'
        ))
        
        # Tambahkan garis harga sekarang
        fig.add_trace(go.Scatter(
            x=methods,
            y=current_prices * len(methods),
            mode='lines',
            name='Harga Sekarang',
            line=dict(color='#d62728', width=3, dash='dash'),
            hovertemplate='Harga Sekarang: Rp %{y:,.0f}<extra></extra>'
        ))
        
        # Konfigurasi layout
        fig.update_layout(
            title='Perbandingan Valuasi Saham',
            xaxis_title='Metode Valuasi',
            yaxis_title='Harga (Rp)',
            template='plotly_white',
            hovermode='x unified',
            height=500,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk prediksi menggunakan Prophet
def predict_with_prophet(ticker, months=6):
    try:
        # Dapatkan data historis
        data = get_stock_data(ticker, period="5y")
        if data.empty or len(data) < 100:
            return None, None, "Data historis tidak cukup untuk prediksi"
        
        # Siapkan data untuk Prophet
        df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)  # Hapus timezone
        
        # Buat dan latih model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df)
        
        # Buat dataframe untuk prediksi
        future = model.make_future_dataframe(periods=30*months, freq='D')
        forecast = model.predict(future)
        
        # Ambil bagian prediksi saja
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30*months)
        
        return forecast, "Sukses"
    
    except Exception as e:
        print(f"Error in Prophet prediction: {str(e)}")
        return None, f"Error: {str(e)}"

# Fungsi untuk menampilkan hasil prediksi
def show_prediction_results(ticker, model_type, months):
    with st.spinner(f'Memprediksi harga {ticker} untuk {months} bulan ke depan...'):
        if model_type == "LSTM":
            predictions, dates, status = predict_with_lstm(ticker, months)
            if predictions is None:
                st.error(f"Prediksi gagal: {status}")
                return
            
            # Buat DataFrame hasil prediksi
            pred_df = pd.DataFrame({
                'Date': dates,
                'Predicted': predictions
            })
            
            # Ambil data historis
            hist_data = get_stock_data(ticker, period="1y")
            
            # Buat plot
            fig = go.Figure()
            
            # Data historis
            fig.add_trace(go.Scatter(
                x=hist_data.index, 
                y=hist_data['Close'],
                mode='lines',
                name='Harga Historis',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Data prediksi
            fig.add_trace(go.Scatter(
                x=pred_df['Date'], 
                y=pred_df['Predicted'],
                mode='lines',
                name='Prediksi LSTM',
                line=dict(color='green', width=2, dash='dot')
            ))
            
            # Konfigurasi layout
            fig.update_layout(
                title=f'Prediksi Harga {ticker} ({months} Bulan) - LSTM',
                xaxis_title='Tanggal',
                yaxis_title='Harga (Rp)',
                template='plotly_white',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan nilai prediksi akhir
            last_pred = pred_df['Predicted'].iloc[-1]
            current_price = get_current_price(ticker)
            change_pct = ((last_pred - current_price) / current_price * 100) if current_price > 0 else 0
            
            st.metric("Prediksi Harga Akhir", 
                     f"Rp {last_pred:,.0f}", 
                     f"{change_pct:.2f}% dari harga sekarang")
            
        else:  # Prophet
            forecast, status = predict_with_prophet(ticker, months)
            if forecast is None:
                st.error(f"Prediksi gagal: {status}")
                return
            
            # Ambil data historis
            hist_data = get_stock_data(ticker, period="1y")
            
            # Buat plot
            fig = go.Figure()
            
            # Data historis
            fig.add_trace(go.Scatter(
                x=hist_data.index, 
                y=hist_data['Close'],
                mode='lines',
                name='Harga Historis',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Data prediksi
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat'],
                mode='lines',
                name='Prediksi Prophet',
                line=dict(color='green', width=2)
            ))
            
            # Interval keyakinan
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                name='Batas Atas'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='Batas Bawah'
            ))
            
            # Konfigurasi layout
            fig.update_layout(
                title=f'Prediksi Harga {ticker} ({months} Bulan) - Prophet',
                xaxis_title='Tanggal',
                yaxis_title='Harga (Rp)',
                template='plotly_white',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan nilai prediksi akhir
            last_pred = forecast['yhat'].iloc[-1]
            current_price = get_current_price(ticker)
            change_pct = ((last_pred - current_price) / current_price * 100) if current_price > 0 else 0
            
            st.metric("Prediksi Harga Akhir", 
                     f"Rp {last_pred:,.0f}", 
                     f"{change_pct:.2f}% dari harga sekarang")

# ========================================================
# FUNGSI BARU: ANALISIS FUNDAMENTAL & RASIO KEUANGAN
# ========================================================

def get_financial_ratios(ticker):
    """Mendapatkan data rasio keuangan dan fundamental saham dari Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Data harga dan volume
        current_price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
        prev_close = info.get('previousClose', np.nan)
        market_cap = info.get('marketCap', np.nan)
        volume = info.get('volume', np.nan)
        avg_volume = info.get('averageVolume', np.nan)
        
        # Rasio profitabilitas
        roe = info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan
        roa = info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan
        npm = info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan
        gross_margin = info.get('grossMargins', np.nan) * 100 if info.get('grossMargins') else np.nan
        operating_margin = info.get('operatingMargins', np.nan) * 100 if info.get('operatingMargins') else np.nan
        
        # Rasio valuasi
        pe_ratio = info.get('trailingPE', np.nan)
        pb_ratio = info.get('priceToBook', np.nan)
        ps_ratio = info.get('priceToSalesTrailing12Months', np.nan)
        ev_ebitda = info.get('enterpriseToEbitda', np.nan)
        
        # Rasio solvabilitas
        der = info.get('debtToEquity', np.nan) * 100 if info.get('debtToEquity') else np.nan
        current_ratio = info.get('currentRatio', np.nan)
        quick_ratio = info.get('quickRatio', np.nan)
        interest_coverage = info.get('earningsBeforeInterestTaxes', np.nan)
        
        # Rasio kinerja
        eps = info.get('trailingEps', info.get('epsTrailingTwelveMonths', np.nan))
        revenue_growth = info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan
        earnings_growth = info.get('earningsGrowth', np.nan) * 100 if info.get('earningsGrowth') else np.nan
        dividend_yield = info.get('dividendYield', np.nan) * 100 if info.get('dividendYield') else np.nan
        payout_ratio = info.get('payoutRatio', np.nan) * 100 if info.get('payoutRatio') else np.nan
        
        # Data fundamental
        book_value = info.get('bookValue', np.nan)
        total_revenue = info.get('totalRevenue', np.nan)
        net_income = info.get('netIncomeToCommon', np.nan)
        total_debt = info.get('totalDebt', np.nan)
        total_equity = info.get('totalStockholderEquity', np.nan)
        free_cash_flow = info.get('freeCashflow', np.nan)
        operating_cash_flow = info.get('operatingCashflow', np.nan)
        
        # Dapatkan data historis rasio
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Siapkan data historis
        historical_ratios = {}
        
        # Historis ROE
        if not financials.empty and not balance_sheet.empty:
            net_income_hist = financials.loc['Net Income'] if 'Net Income' in financials.index else None
            equity_hist = balance_sheet.loc['Total Stockholder Equity'] if 'Total Stockholder Equity' in balance_sheet.index else None
            
            if net_income_hist is not None and equity_hist is not None:
                roe_hist = (net_income_hist / equity_hist) * 100
                historical_ratios['ROE'] = roe_hist
                
        # Historis DER
        if not balance_sheet.empty:
            total_debt_hist = balance_sheet.loc['Total Debt'] if 'Total Debt' in balance_sheet.index else None
            total_equity_hist = balance_sheet.loc['Total Stockholder Equity'] if 'Total Stockholder Equity' in balance_sheet.index else None
            
            if total_debt_hist is not None and total_equity_hist is not None:
                der_hist = (total_debt_hist / total_equity_hist) * 100
                historical_ratios['DER'] = der_hist
        
        # Historis NPM
        if not financials.empty:
            net_income_hist = financials.loc['Net Income'] if 'Net Income' in financials.index else None
            revenue_hist = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
            
            if net_income_hist is not None and revenue_hist is not None:
                npm_hist = (net_income_hist / revenue_hist) * 100
                historical_ratios['NPM'] = npm_hist
                
        # Historis EPS
        if not financials.empty:
            eps_hist = financials.loc['Diluted EPS'] if 'Diluted EPS' in financials.index else None
            if eps_hist is not None:
                historical_ratios['EPS'] = eps_hist
                
        # Historis Revenue Growth
        if not financials.empty:
            revenue_hist = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
            if revenue_hist is not None:
                revenue_growth_hist = revenue_hist.pct_change(periods=-1) * 100
                historical_ratios['Revenue Growth'] = revenue_growth_hist
        
        return {
            'current_price': current_price,
            'prev_close': prev_close,
            'market_cap': market_cap,
            'volume': volume,
            'avg_volume': avg_volume,
            'roe': roe,
            'roa': roa,
            'npm': npm,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'ps_ratio': ps_ratio,
            'ev_ebitda': ev_ebitda,
            'der': der,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'interest_coverage': interest_coverage,
            'eps': eps,
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'dividend_yield': dividend_yield,
            'payout_ratio': payout_ratio,
            'book_value': book_value,
            'total_revenue': total_revenue,
            'net_income': net_income,
            'total_debt': total_debt,
            'total_equity': total_equity,
            'free_cash_flow': free_cash_flow,
            'operating_cash_flow': operating_cash_flow,
            'historical_ratios': historical_ratios
        }
    except Exception as e:
        st.error(f"Error mendapatkan data fundamental untuk {ticker}: {str(e)}")
        return {}

def display_fundamental_analysis(ticker, financial_data):
    """Menampilkan analisis fundamental saham"""
    st.subheader(f"Analisis Fundamental {ticker}")
    
    # Tampilkan metrik utama
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Harga", 
               f"Rp {financial_data['current_price']:,.0f}", 
               f"{((financial_data['current_price'] - financial_data['prev_close'])/financial_data['prev_close']*100):.2f}%" if not np.isnan(financial_data['prev_close']) else "N/A")
    col1.metric("Kapitalisasi Pasar", f"Rp {financial_data['market_cap']:,.2f}" if not np.isnan(financial_data['market_cap']) else "N/A")
    
    col2.metric("Volume", f"{financial_data['volume']:,.0f}", 
               f"{(financial_data['volume']/financial_data['avg_volume']*100 if not np.isnan(financial_data['avg_volume']) and financial_data['avg_volume'] > 0 else 0):.0f}% vs rata-rata" if not np.isnan(financial_data['volume']) else "N/A")
    col2.metric("EPS", f"Rp {financial_data['eps']:,.0f}" if not np.isnan(financial_data['eps']) else "N/A")
    
    col3.metric("ROE", f"{financial_data['roe']:.2f}%" if not np.isnan(financial_data['roe']) else "N/A", 
               ">15% bagus" if not np.isnan(financial_data['roe']) and financial_data['roe'] > 15 else "<10% kurang")
    col3.metric("DER", f"{financial_data['der']:.2f}%" if not np.isnan(financial_data['der']) else "N/A", 
               "<80% sehat" if not np.isnan(financial_data['der']) and financial_data['der'] < 80 else ">100% berisiko")
    
    col4.metric("NPM", f"{financial_data['npm']:.2f}%" if not np.isnan(financial_data['npm']) else "N/A", 
               ">20% bagus" if not np.isnan(financial_data['npm']) and financial_data['npm'] > 20 else "<10% kurang")
    col4.metric("Dividen Yield", f"{financial_data['dividend_yield']:.2f}%" if not np.isnan(financial_data['dividend_yield']) else "N/A")
    
    # Tab untuk berbagai kategori rasio
    tab_ratios, tab_fundamental, tab_hist = st.tabs(["Rasio Keuangan", "Data Fundamental", "Analisis Historis"])
    
    with tab_ratios:
        st.subheader("Rasio Keuangan Utama")
        
        # Buat DataFrame untuk rasio
        ratio_data = {
            'Kategori': ['Profitabilitas', 'Profitabilitas', 'Profitabilitas', 'Profitabilitas', 
                         'Valuasi', 'Valuasi', 'Valuasi', 'Valuasi',
                         'Solvabilitas', 'Solvabilitas', 'Solvabilitas', 'Solvabilitas',
                         'Kinerja', 'Kinerja', 'Kinerja', 'Kinerja'],
            'Rasio': ['ROE', 'ROA', 'NPM', 'Gross Margin',
                      'PER', 'PBV', 'PSR', 'EV/EBITDA',
                      'DER', 'Current Ratio', 'Quick Ratio', 'Interest Coverage',
                      'EPS', 'Revenue Growth', 'Earnings Growth', 'Dividend Yield'],
            'Nilai': [financial_data['roe'], financial_data['roa'], financial_data['npm'], financial_data['gross_margin'],
                      financial_data['pe_ratio'], financial_data['pb_ratio'], financial_data['ps_ratio'], financial_data['ev_ebitda'],
                      financial_data['der'], financial_data['current_ratio'], financial_data['quick_ratio'], financial_data['interest_coverage'],
                      financial_data['eps'], financial_data['revenue_growth'], financial_data['earnings_growth'], financial_data['dividend_yield']],
            'Interpretasi': [
                'Return on Equity > 15% bagus',
                'Return on Assets > 5% bagus',
                'Net Profit Margin > 20% bagus',
                'Gross Margin > 40% bagus',
                'PER rendah lebih baik',
                'PBV < 1 potensi undervalued',
                'PSR < 1 bagus',
                'EV/EBITDA < 10 bagus',
                'DER < 80% sehat',
                'Current Ratio > 1.5 bagus',
                'Quick Ratio > 1 bagus',
                'Interest Coverage > 3 bagus',
                'EPS tinggi lebih baik',
                'Revenue Growth positif bagus',
                'Earnings Growth positif bagus',
                'Dividend Yield > 3% bagus'
            ]
        }
        
        ratio_df = pd.DataFrame(ratio_data)
        
        # Tampilkan tabel dengan warna
        def color_ratios(val):
            if val == 'ROE' and not np.isnan(financial_data['roe']):
                return 'background-color: lightgreen' if financial_data['roe'] > 15 else 'background-color: salmon'
            elif val == 'DER' and not np.isnan(financial_data['der']):
                return 'background-color: lightgreen' if financial_data['der'] < 80 else 'background-color: salmon'
            elif val == 'NPM' and not np.isnan(financial_data['npm']):
                return 'background-color: lightgreen' if financial_data['npm'] > 20 else 'background-color: salmon'
            elif val == 'PER' and not np.isnan(financial_data['pe_ratio']):
                return 'background-color: lightgreen' if financial_data['pe_ratio'] < 20 else 'background-color: salmon'
            return ''
        
        st.dataframe(
            ratio_df.style.applymap(color_ratios, subset=['Rasio']).format({
                'Nilai': '{:.2f}'
            }),
            height=600
        )
    
    with tab_fundamental:
        st.subheader("Data Fundamental")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Neraca Keuangan")
            st.metric("Total Aset", f"Rp {financial_data['total_equity'] + financial_data['total_debt']:,.0f}" if not np.isnan(financial_data['total_equity']) and not np.isnan(financial_data['total_debt']) else "N/A")
            st.metric("Total Ekuitas", f"Rp {financial_data['total_equity']:,.0f}" if not np.isnan(financial_data['total_equity']) else "N/A")
            st.metric("Total Hutang", f"Rp {financial_data['total_debt']:,.0f}" if not np.isnan(financial_data['total_debt']) else "N/A")
            st.metric("Nilai Buku per Saham", f"Rp {financial_data['book_value']:,.0f}" if not np.isnan(financial_data['book_value']) else "N/A")
        
        with col2:
            st.write("### Laporan Laba Rugi")
            st.metric("Pendapatan", f"Rp {financial_data['total_revenue']:,.0f}" if not np.isnan(financial_data['total_revenue']) else "N/A")
            st.metric("Laba Bersih", f"Rp {financial_data['net_income']:,.0f}" if not np.isnan(financial_data['net_income']) else "N/A")
            st.metric("Arus Kas Operasi", f"Rp {financial_data['operating_cash_flow']:,.0f}" if not np.isnan(financial_data['operating_cash_flow']) else "N/A")
            st.metric("Arus Kas Bebas", f"Rp {financial_data['free_cash_flow']:,.0f}" if not np.isnan(financial_data['free_cash_flow']) else "N/A")
    
    with tab_hist:
        st.subheader("Analisis Historis Rasio")
        
        # Pilih rasio untuk ditampilkan
        selected_ratio = st.selectbox("Pilih Rasio untuk Analisis Historis", 
                                     ['ROE', 'DER', 'NPM', 'EPS', 'Revenue Growth'])
        
        if financial_data['historical_ratios'].get(selected_ratio) is not None:
            ratio_hist = financial_data['historical_ratios'][selected_ratio]
            
            # Buat DataFrame
            hist_df = pd.DataFrame({
                'Tahun': ratio_hist.index.year,
                'Nilai': ratio_hist.values
            })
            
            # Tampilkan grafik
            fig = px.line(
                hist_df, 
                x='Tahun', 
                y='Nilai',
                title=f'Perkembangan {selected_ratio} Historis',
                markers=True,
                text='Nilai'
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(
                yaxis_title=selected_ratio,
                template='plotly_white',
                hovermode='x unified',
                height=500
            )
            
            # Tampilkan rata-rata
            avg_value = ratio_hist.mean()
            last_value = ratio_hist.iloc[0]
            
            fig.add_hline(
                y=avg_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Rata-rata: {avg_value:.2f}",
                annotation_position="bottom right"
            )
            
            # Analisis tren
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretasi tren
            if len(ratio_hist) > 1:
                trend = "Meningkat" if last_value > ratio_hist.iloc[1] else "Menurun"
                st.write(f"**Tren {selected_ratio}:** {trend} ({last_value:.2f} vs {ratio_hist.iloc[1]:.2f} tahun sebelumnya)")
                
                if selected_ratio == 'ROE':
                    st.info("ROE yang stabil di atas 15% menunjukkan efisiensi penggunaan modal yang baik")
                elif selected_ratio == 'DER':
                    st.info("DER di bawah 80% menunjukkan struktur modal yang sehat")
                elif selected_ratio == 'NPM':
                    st.info("NPM di atas 20% menunjukkan kemampuan menghasilkan laba yang kuat")
                elif selected_ratio == 'EPS':
                    st.info("EPS yang meningkat menunjukkan pertumbuhan laba per saham")
                elif selected_ratio == 'Revenue Growth':
                    st.info("Pertumbuhan pendapatan positif menunjukkan ekspansi bisnis")
            else:
                st.warning("Data historis tidak cukup untuk analisis tren")
        else:
            st.warning(f"Data historis untuk {selected_ratio} tidak tersedia")
    
    # Analisis kesehatan keuangan
    st.subheader("Analisis Kesehatan Keuangan")
    col1, col2, col3 = st.columns(3)
    
    # Profitabilitas
    profit_score = 0
    if not np.isnan(financial_data['roe']) and financial_data['roe'] > 15:
        profit_score += 1
    if not np.isnan(financial_data['npm']) and financial_data['npm'] > 20:
        profit_score += 1
    if not np.isnan(financial_data['gross_margin']) and financial_data['gross_margin'] > 40:
        profit_score += 1
    
    col1.metric("Profitabilitas", f"{profit_score}/3", 
               "Sangat Baik" if profit_score >= 2 else "Cukup" if profit_score == 1 else "Kurang")
    
    # Solvabilitas
    solv_score = 0
    if not np.isnan(financial_data['der']) and financial_data['der'] < 80:
        solv_score += 1
    if not np.isnan(financial_data['current_ratio']) and financial_data['current_ratio'] > 1.5:
        solv_score += 1
    if not np.isnan(financial_data['interest_coverage']) and financial_data['interest_coverage'] > 3:
        solv_score += 1
    
    col2.metric("Solvabilitas", f"{solv_score}/3", 
               "Sangat Sehat" if solv_score >= 2 else "Cukup" if solv_score == 1 else "Berisiko")
    
    # Pertumbuhan
    growth_score = 0
    if not np.isnan(financial_data['revenue_growth']) and financial_data['revenue_growth'] > 10:
        growth_score += 1
    if not np.isnan(financial_data['earnings_growth']) and financial_data['earnings_growth'] > 10:
        growth_score += 1
    if not np.isnan(financial_data['eps']) and financial_data['eps'] > 0:
        growth_score += 1
    
    col3.metric("Pertumbuhan", f"{growth_score}/3", 
               "Kuat" if growth_score >= 2 else "Sedang" if growth_score == 1 else "Lemah")
    
    # Rekomendasi keseluruhan
    total_score = profit_score + solv_score + growth_score
    if total_score >= 7:
        st.success("**REKOMENDASI: BELI** - Kesehatan keuangan sangat baik dengan skor 8-9")
    elif total_score >= 5:
        st.warning("**REKOMENDASI: TAHAN** - Kesehatan keuangan cukup baik dengan skor 5-7")
    else:
        st.error("**REKOMENDASI: JUAL** - Kesehatan keuangan kurang baik dengan skor di bawah 5")
    
    st.write(f"**Total Skor Kesehatan Keuangan:** {total_score}/9")

# ========================================================
# TAMPILAN STREAMLIT
# ========================================================

# Tampilan Streamlit
st.title("📈 Analisis Portofolio Saham & Fundamental")

# Buat tab untuk navigasi
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analisis Portofolio", "Simulasi Pensiun", "Prediksi Harga", "Valuasi Saham", "Analisis Fundamental"])

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
                            key=lambda x: abs(x[1] - row['Current Price']))
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


# ========================================================
# TAB BARU: PREDIKSI HARGA SAHAM
# ========================================================
with tab3:
    st.header("🔮 Prediksi Harga Saham Jangka Menengah")
    st.write("""
    Prediksi harga saham untuk 1-6 bulan ke depan menggunakan model:
    - **LSTM**: Long Short-Term Memory (model deep learning khusus data time series)
    - **Prophet**: Model forecasting time series dari Facebook
    
    Catatan:
    - Prediksi bersifat indikatif dan tidak menjamin akurasi mutlak
    - Hasil prediksi dapat berbeda-beda tergantung kondisi pasar
    - Gunakan sebagai salah satu referensi pengambilan keputusan
    """)
    
    # Pilihan saham dan parameter
    col1, col2 = st.columns(2)
    with col1:
        # Pilih indeks saham
        index_selection = st.selectbox("Pilih Indeks Saham", ["Kompas100", "LQ45"])
        
        # Pilih saham berdasarkan indeks
        stocks = KOMPAS100 if index_selection == "Kompas100" else LQ45
        selected_stock = st.selectbox("Pilih Saham", stocks)
        
    with col2:
        # Pilih model prediksi
        model_type = st.selectbox("Pilih Model Prediksi", ["LSTM", "Prophet"])
        
        # Pilih periode prediksi
        months = st.slider("Periode Prediksi (Bulan)", 1, 6, 3)
    
    # Tombol prediksi
    predict_button = st.button("Mulai Prediksi")
    
    if predict_button and selected_stock:
        ticker = f"{selected_stock}.JK"
        show_prediction_results(ticker, model_type, months)
    
    # Informasi tambahan
    st.markdown("---")
    with st.expander("Penjelasan Model Prediksi"):
        st.subheader("LSTM (Long Short-Term Memory)")
        st.write("""
        LSTM adalah jenis jaringan saraf berulang (RNN) yang dirancang khusus untuk memproses data sekuensial dan deret waktu. 
        Keunggulan LSTM:
        - Mampu mempelajari ketergantungan jangka panjang dalam data
        - Menangani pola non-linear dengan baik
        - Tahan terhadap masalah vanishing gradient
        
        Dalam aplikasi ini:
        - Model dilatih menggunakan data 2 tahun terakhir
        - Menggunakan arsitektur 2 lapisan LSTM dengan dropout
        - Melakukan prediksi per hari untuk 1-6 bulan ke depan
        """)
        
        st.subheader("Prophet")
        st.write("""
        Prophet adalah model forecasting time series yang dikembangkan oleh Facebook. Model ini dirancang untuk:
        - Menangani pola musiman (harian, mingguan, tahunan)
        - Memperhitungkan hari libur dan event khusus
        - Robust terhadap missing data dan outliers
        
        Dalam aplikasi ini:
        - Model dilatih menggunakan data 5 tahun terakhir
        - Menambahkan komponen musiman bulanan
        - Menghasilkan prediksi beserta interval keyakinan
        """)
        
        st.subheader("Perbandingan Kedua Model")
        st.write("""
        | Fitur                | LSTM                          | Prophet                       |
        |----------------------|-------------------------------|-------------------------------|
        | Akurasi jangka pendek | Sangat baik                  | Baik                          |
        | Akurasi jangka panjang | Baik                         | Sangat baik                   |
        | Kecepatan pelatihan   | Lambat (perlu GPU)           | Cepat                         |
        | Interpretabilitas     | Rendah (black box)           | Tinggi                        |
        | Penanganan musiman    | Terbatas                     | Sangat baik                   |
        | Data minimal          | ≥100 hari                    | ≥100 hari                     |
        """)

with tab4:
    st.header("💰 Valuasi Harga Wajar Saham")
    st.write("""
    Hitung harga wajar saham menggunakan berbagai metode valuasi:
    - **PBV (Price to Book Value)**: Valuasi berdasarkan nilai buku perusahaan
    - **PER (Price to Earnings Ratio)**: Valuasi berdasarkan kemampuan perusahaan menghasilkan laba
    - **DCF (Discounted Cash Flow)**: Valuasi berdasarkan arus kas masa depan yang didiskontokan
    - **Graham Formula**: Metode klasik dari Benjamin Graham (Bapak Value Investing)
    """)
    
    # Pilih indeks saham
    col1, col2 = st.columns(2)
    with col1:
        index_selection = st.selectbox("Pilih Indeks Saham", ["Kompas100", "LQ45"], key="valuation_index")
        
        # Pilih saham berdasarkan indeks
        stocks = KOMPAS100 if index_selection == "Kompas100" else LQ45
        selected_stock = st.selectbox("Pilih Saham", stocks, key="valuation_stock")
    
    with col2:
        st.write("### Parameter Umum")
        safety_margin = st.slider("Margin of Safety (%)", 0, 30, 15, key="safety_margin") / 100
        industry_pb = st.number_input("Rata-rata PBV Industri", min_value=0.0, value=2.5, step=0.1, key="industry_pb")
        industry_pe = st.number_input("Rata-rata PER Industri", min_value=0.0, value=15.0, step=0.5, key="industry_pe")
    
    # Dapatkan data fundamental
    ticker = f"{selected_stock}.JK"
    financial_data = get_financial_data(ticker)
    
    if financial_data:
        st.subheader(f"Data Fundamental {selected_stock}")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Harga Sekarang", f"Rp {financial_data['current_price']:,.0f}" if not np.isnan(financial_data['current_price']) else "N/A")
        col1.metric("Nilai Buku per Saham", f"Rp {financial_data['book_value']:,.0f}" if not np.isnan(financial_data['book_value']) else "N/A")
        
        col2.metric("EPS (Laba per Saham)", f"Rp {financial_data['eps']:,.0f}" if not np.isnan(financial_data['eps']) else "N/A")
        col2.metric("Dividen per Saham", f"Rp {financial_data['last_dividend']:,.0f}" if not np.isnan(financial_data['last_dividend']) else "N/A")
        
        col3.metric("ROE (%)", f"{financial_data['roe']:.2f}%" if not np.isnan(financial_data['roe']) else "N/A")
        col3.metric("Beta", f"{financial_data['beta']:.2f}" if not np.isnan(financial_data['beta']) else "N/A")
    
    # Form input untuk setiap metode valuasi
    st.markdown("---")
    st.subheader("Parameter Valuasi")
    
    with st.expander("Metode PBV (Price to Book Value)", expanded=True):
        st.info("""
        **Rumus:**  
        Harga Wajar = Nilai Buku per Saham × PBV Industri  
        Harga Beli Target = Harga Wajar × (1 - Margin of Safety)
        """)
        
        st.write("Gunakan parameter yang sudah ditentukan di atas")
    
    with st.expander("Metode PER (Price to Earnings Ratio)", expanded=True):
        st.info("""
        **Rumus:**  
        PER Wajar = PER Industri × (1 + Tingkat Pertumbuhan)  
        Harga Wajar = EPS × PER Wajar  
        Harga Beli Target = Harga Wajar × (1 - Margin of Safety)
        """)
        
        growth_rate_per = st.slider("Tingkat Pertumbuhan Laba (%)", 0.0, 30.0, 10.0, step=0.5, key="growth_per") / 100
    
    with st.expander("Metode DCF (Discounted Cash Flow)", expanded=True):
        st.info("""
        **Rumus:**  
        1. Proyeksikan arus kas 5 tahun ke depan dengan tingkat pertumbuhan  
        2. Hitung nilai terminal setelah 5 tahun  
        3. Diskontokan semua arus kas ke nilai sekarang  
        4. Harga Wajar = Total nilai sekarang  
        5. Harga Beli Target = Harga Wajar × (1 - Margin of Safety)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            growth_rate_5y = st.slider("Tingkat Pertumbuhan 5 Tahun (%)", 0.0, 30.0, 12.0, step=0.5, key="growth_5y") / 100
            terminal_growth = st.slider("Tingkat Pertumbuhan Terminal (%)", 0.0, 10.0, 3.0, step=0.1, key="terminal_growth") / 100
        with col2:
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.5, step=0.1, key="risk_free") / 100
            market_risk_premium = st.slider("Market Risk Premium (%)", 0.0, 15.0, 5.5, step=0.1, key="risk_premium") / 100
        
        # Hitung discount rate dengan CAPM
        if not np.isnan(financial_data.get('beta', np.nan)):
            beta = financial_data['beta']
            discount_rate = risk_free_rate + beta * market_risk_premium
            st.write(f"**Discount Rate (CAPM):** {discount_rate*100:.2f}% = {risk_free_rate*100:.2f}% + {beta:.2f} × {market_risk_premium*100:.2f}%")
        else:
            discount_rate = st.slider("Discount Rate (%)", 0.0, 30.0, 10.0, step=0.5, key="discount_rate") / 100
    
    with st.expander("Metode Graham Formula", expanded=True):
        st.info("""
        **Rumus Benjamin Graham:**  
        Harga Wajar = EPS × (8.5 + 2 × Tingkat Pertumbuhan 7 Tahun) × (4.4 / Yield Obligasi AAA)  
        Harga Beli Target = Harga Wajar × 0.75 (Margin of Safety 25%)
        """)
        
        growth_rate_7y = st.slider("Tingkat Pertumbuhan 7 Tahun (%)", 0.0, 30.0, 10.0, step=0.5, key="growth_7y") / 100
        bond_yield = st.slider("Yield Obligasi AAA (%)", 0.0, 15.0, 7.0, step=0.1, key="bond_yield") / 100
    
    # Tombol hitung valuasi
    if st.button("Hitung Valuasi", key="calculate_valuation"):
        if financial_data:
            valuation_results = {}
            
            # PBV Valuation
            if not np.isnan(financial_data['book_value']):
                fair_value, buy_price = pbv_valuation(
                    financial_data['book_value'], 
                    industry_pb, 
                    safety_margin
                )
                valuation_results['PBV'] = (fair_value, buy_price)
            
            # PER Valuation
            if not np.isnan(financial_data['eps']):
                fair_value, buy_price = per_valuation(
                    financial_data['eps'], 
                    industry_pe, 
                    growth_rate_per,
                    safety_margin
                )
                valuation_results['PER'] = (fair_value, buy_price)
            
            # DCF Valuation
            if not np.isnan(financial_data['eps']):
                fair_value, buy_price = dcf_valuation(
                    financial_data['eps'], 
                    growth_rate_5y,
                    terminal_growth,
                    discount_rate,
                    safety_margin
                )
                valuation_results['DCF'] = (fair_value, buy_price)
            
            # Graham Valuation
            if not np.isnan(financial_data['eps']) and not np.isnan(financial_data['book_value']):
                fair_value, buy_price = graham_valuation(
                    financial_data['eps'], 
                    financial_data['book_value'],
                    growth_rate_7y,
                    bond_yield
                )
                valuation_results['Graham'] = (fair_value, buy_price)
            
            # Tampilkan hasil
            if valuation_results:
                valuation_data = {
                    'current_price': financial_data['current_price'],
                    'results': valuation_results
                }
                display_valuation_results(ticker, valuation_data)
            else:
                st.error("Tidak ada metode valuasi yang dapat dihitung karena data tidak lengkap")
        else:
            st.error("Tidak dapat melakukan valuasi karena data fundamental tidak tersedia")
    
    # Penjelasan metode valuasi
    st.markdown("---")
    with st.expander("📚 Penjelasan Metode Valuasi"):
        st.subheader("PBV (Price to Book Value)")
        st.write("""
        **Konsep:**  
        Metode PBV membandingkan harga pasar saham dengan nilai buku perusahaan. 
        Nilai buku dihitung dari total aset dikurangi total kewajiban, kemudian dibagi jumlah saham.
        
        **Cara Interpretasi:**
        - PBV < 1: Saham dinilai di bawah nilai bukunya (potensi undervalued)
        - PBV 1-2: Saham dinilai wajar
        - PBV > 2: Saham dinilai di atas nilai bukunya (potensi overvalued)
        
        **Kelebihan:**
        - Berguna untuk perusahaan dengan aset berwujud besar (perbankan, properti)
        - Lebih stabil dibanding metode berbasis laba
        
        **Kekurangan:**
        - Kurang relevan untuk perusahaan berbasis intelektual (teknologi)
        - Tidak memperhitungkan potensi pertumbuhan
        """)
        
        st.subheader("PER (Price to Earnings Ratio)")
        st.write("""
        **Konsep:**  
        Metode PER membandingkan harga saham dengan laba per saham (EPS). 
        Rasio ini menunjukkan berapa banyak investor bersedia membayar untuk setiap rupiah laba perusahaan.
        
        **Cara Interpretasi:**
        - PER < industri: Potensi undervalued
        - PER ≈ industri: Nilai wajar
        - PER > industri: Potensi overvalued
        
        **Kelebihan:**
        - Mudah dihitung dan dipahami
        - Memperhitungkan profitabilitas perusahaan
        
        **Kekurangan:**
        - Sensitif terhadap manipulasi akuntansi
        - Tidak berguna untuk perusahaan yang belum profit
        """)
        
        st.subheader("DCF (Discounted Cash Flow)")
        st.write("""
        **Konsep:**  
        Metode DCF menghitung nilai intrinsik saham dengan mendiskontokan arus kas masa depan ke nilai sekarang. 
        Metode ini terdiri dari dua bagian utama:
        1. Proyeksi arus kas bebas selama 5-10 tahun
        2. Perhitungan nilai terminal setelah periode proyeksi
        
        **Cara Interpretasi:**
        - Nilai intrinsik > harga pasar: Potensi undervalued
        - Nilai intrinsik < harga pasar: Potensi overvalued
        
        **Kelebihan:**
        - Mempertimbangkan nilai waktu uang
        - Fokus pada arus kas (lebih sulit dimanipulasi)
        
        **Kekurangan:**
        - Sangat bergantung pada asumsi pertumbuhan dan diskonto
        - Kompleks dan banyak variabel
        """)
        
        st.subheader("Graham Formula (Benjamin Graham)")
        st.write("""
        **Konsep:**  
        Dikembangkan oleh Benjamin Graham (guru Warren Buffett), formula ini dirancang untuk investor nilai. 
        Formula dasar Graham:
        ```
        Harga Wajar = EPS × (8.5 + 2g) × (4.4 / AAA Bond Yield)
        ```
        Dimana:
        - EPS: Laba per saham
        - g: Tingkat pertumbuhan laba 7-10 tahun
        - AAA Bond Yield: Yield obligasi pemerintah AAA
        
        **Cara Interpretasi:**
        - Harga pasar < 80% nilai wajar: Potensi beli
        - Harga pasar > 120% nilai wajar: Potensi jual
        
        **Kelebihan:**
        - Sederhana dan mudah dihitung
        - Memasukkan faktor pertumbuhan dan suku bunga
        
        **Kekurangan:**
        - Dirancang untuk pasar yang berbeda era Graham
        - Kurang akurat untuk perusahaan pertumbuhan tinggi
        """)
    
    st.markdown("---")
    st.info("⚠️ **Peringatan Investasi**: Valuasi saham merupakan perkiraan berdasarkan asumsi dan data historis. "
            "Hasil valuasi tidak menjamin kinerja saham di masa depan. Selalu lakukan riset lebih lanjut sebelum berinvestasi.")
    
    st.markdown("---")
    st.info("🚨 **Peringatan Investasi**: Prediksi harga saham bersifat spekulatif dan tidak dapat dijadikan satu-satunya acuan pengambilan keputusan investasi. Selalu lakukan analisis fundamental dan pertimbangkan risiko investasi.")

with tab5:
    st.header("📊 Analisis Fundamental Saham")
    st.write("""
    Analisis rasio keuangan dan fundamental perusahaan:
    - **Profitabilitas**: ROE, ROA, NPM, Gross Margin
    - **Solvabilitas**: DER, Current Ratio, Quick Ratio
    - **Valuasi**: PER, PBV, PSR, EV/EBITDA
    - **Kinerja**: EPS, Pertumbuhan Pendapatan, Pertumbuhan Laba
    - **Dividen**: Dividend Yield, Payout Ratio
    """)
    
    # Pilih indeks saham
    col1, col2 = st.columns(2)
    with col1:
        index_selection = st.selectbox("Pilih Indeks Saham", ["Kompas100", "LQ45"], key="fundamental_index")
        
        # Pilih saham berdasarkan indeks
        stocks = KOMPAS100 if index_selection == "Kompas100" else LQ45
        selected_stock = st.selectbox("Pilih Saham", stocks, key="fundamental_stock")
    
    with col2:
        st.write("### Benchmark Industri")
        industry_roe = st.number_input("ROE Rata-rata Industri (%)", min_value=0.0, value=15.0, step=0.5, key="industry_roe")
        industry_der = st.number_input("DER Rata-rata Industri (%)", min_value=0.0, value=80.0, step=1.0, key="industry_der")
        industry_npm = st.number_input("NPM Rata-rata Industri (%)", min_value=0.0, value=20.0, step=0.5, key="industry_npm")
    
    # Dapatkan data fundamental
    ticker = f"{selected_stock}.JK"
    financial_data = get_financial_ratios(ticker)
    
    if financial_data:
        display_fundamental_analysis(ticker, financial_data)
    else:
        st.error("Tidak dapat mendapatkan data fundamental untuk saham ini")
    
    # Penjelasan rasio keuangan
    st.markdown("---")
    with st.expander("📚 Penjelasan Rasio Keuangan"):
        st.subheader("ROE (Return on Equity)")
        st.write("""
        **Rumus:**  
        ROE = (Laba Bersih / Ekuitas Pemegang Saham) × 100%  
        
        **Interpretasi:**  
        Mengukur efisiensi penggunaan modal sendiri.  
        - ROE > 15%: Sangat baik  
        - ROE 10-15%: Baik  
        - ROE < 10%: Kurang  
        
        **Kelebihan:**  
        - Indikator utama profitabilitas  
        - Mudah dibandingkan antar perusahaan  
        
        **Kekurangan:**  
        - Dapat dimanipulasi dengan mengurangi ekuitas  
        - Tidak memperhitungkan hutang  
        """)
        
        st.subheader("DER (Debt to Equity Ratio)")
        st.write("""
        **Rumus:**  
        DER = (Total Hutang / Ekuitas Pemegang Saham) × 100%  
        
        **Interpretasi:**  
        Mengukur proporsi hutang terhadap ekuitas.  
        - DER < 80%: Sehat  
        - DER 80-150%: Hati-hati  
        - DER > 150%: Berisiko tinggi  
        
        **Kelebihan:**  
        - Menunjukkan risiko finansial perusahaan  
        - Indikator struktur modal  
        
        **Kekurangan:**  
        - Tidak membedakan jenis hutang  
        - Industri berbeda memiliki standar berbeda  
        """)
        
        st.subheader("NPM (Net Profit Margin)")
        st.write("""
        **Rumus:**  
        NPM = (Laba Bersih / Pendapatan) × 100%  
        
        **Interpretasi:**  
        Mengukur persentase laba dari pendapatan.  
        - NPM > 20%: Sangat efisien  
        - NPM 10-20%: Baik  
        - NPM < 10%: Kurang  
        
        **Kelebihan:**  
        - Menunjukkan efisiensi operasional  
        - Dapat dibandingkan antar industri  
        
        **Kekurangan:**  
        - Tidak memperhitungkan struktur modal  
        - Dapat dipengaruhi oleh biaya non-operasional  
        """)
        
        st.subheader("EPS (Earnings Per Share)")
        st.write("""
        **Rumus:**  
        EPS = Laba Bersih / Jumlah Saham Beredar  
        
        **Interpretasi:**  
        Mengukur laba yang dihasilkan per lembar saham.  
        - EPS tinggi: Lebih baik  
        - EPS meningkat: Pertumbuhan baik  
        
        **Kelebihan:**  
        - Langsung terkait dengan nilai pemegang saham  
        - Dasar perhitungan PER  
        
        **Kekurangan:**  
        - Tidak memperhitungkan struktur modal  
        - Dapat dimanipulasi dengan pembelian kembali saham  
        """)
        
        st.subheader("Pertumbuhan Pendapatan")
        st.write("""
        **Rumus:**  
        Pertumbuhan = ((Pendapatan Tahun Ini - Pendapatan Tahun Lalu) / Pendapatan Tahun Lalu) × 100%  
        
        **Interpretasi:**  
        Mengukur pertumbuhan bisnis perusahaan.  
        - >20%: Pertumbuhan tinggi  
        - 10-20%: Pertumbuhan sedang  
        - <10%: Pertumbuhan rendah  
        
        **Kelebihan:**  
        - Indikator utama ekspansi bisnis  
        - Prospek masa depan perusahaan  
        
        **Kekurangan:**  
        - Tidak menunjukkan profitabilitas  
        - Dapat dipengaruhi oleh akuisisi  
        """)
    
    st.markdown("---")
    st.info("⚠️ **Peringatan Analisis**: Rasio keuangan hanyalah salah satu alat analisis. "
            "Selalu pertimbangkan faktor kualitatif dan kondisi industri sebelum membuat keputusan investasi.")
