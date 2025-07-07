import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging
import warnings

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter peringatan
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.fetch_data()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.forecast_days = 30
        
    def fetch_data(self):
        """Mengambil data historis saham dengan volume lebih tinggi"""
        try:
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period="5y", interval="1d")
            
            if hist.empty:
                logger.warning(f"Tidak ada data untuk {self.ticker}")
                return pd.DataFrame()
            
            # Hitung moving averages
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA100'] = hist['Close'].rolling(window=100).mean()
            
            # Hitung volatilitas
            hist['Volatility'] = hist['Close'].pct_change().rolling(window=14).std()
            
            # Hitung RSI
            hist['RSI'] = self.calculate_rsi(hist['Close'])
            
            # Hitung MACD
            hist['MACD'], hist['MACD_Signal'] = self.calculate_macd(hist['Close'])
            
            # Hapus baris dengan nilai NaN
            hist = hist.dropna()
            
            return hist
        except Exception as e:
            logger.error(f"Error mengambil data {self.ticker}: {e}")
            return pd.DataFrame()
    
    def predict_trend(self):
        """Memprediksi trend dengan ensemble model"""
        if self.data.empty:
            return "Netral", 0, {}
        
        try:
            # Prediksi dengan berbagai model
            arima_pred, arima_metrics = self.predict_with_arima()
            lstm_pred, lstm_metrics = self.predict_with_lstm()
            rf_pred, rf_metrics = self.predict_with_random_forest()
            
            # Gabungkan prediksi (rata-rata tertimbang)
            weights = {
                'arima': 0.3,
                'lstm': 0.4,
                'rf': 0.3
            }
            ensemble_pred = (
                weights['arima'] * arima_pred + 
                weights['lstm'] * lstm_pred + 
                weights['rf'] * rf_pred
            )
            
            # Tentukan trend berdasarkan prediksi ensemble
            last_price = self.data['Close'].iloc[-1]
            price_change = ensemble_pred - last_price
            change_pct = price_change / last_price * 100
            
            if change_pct > 3:
                trend = "Kuat Naik"
            elif change_pct > 1:
                trend = "Naik"
            elif change_pct < -3:
                trend = "Kuat Turun"
            elif change_pct < -1:
                trend = "Turun"
            else:
                trend = "Netral"
            
            # Kumpulkan metrik evaluasi
            metrics = {
                'arima': arima_metrics,
                'lstm': lstm_metrics,
                'rf': rf_metrics,
                'ensemble': {
                    'predicted_price': ensemble_pred,
                    'change_pct': change_pct
                }
            }
            
            return trend, ensemble_pred, metrics
        except Exception as e:
            logger.error(f"Error memprediksi trend: {e}")
            return "Netral", self.data['Close'].iloc[-1], {}

    def predict_with_arima(self):
        """Prediksi menggunakan model ARIMA"""
        try:
            # Siapkan data
            data = self.data['Close'].values
            
            # Bagi data training/testing
            train_size = int(len(data) * 0.8)
            train, test = data[:train_size], data[train_size:]
            
            # Model ARIMA
            model = ARIMA(train, order=(5,1,0))
            model_fit = model.fit()
            
            # Prediksi
            forecast = model_fit.forecast(steps=len(test))
            
            # Evaluasi
            mse = mean_squared_error(test, forecast)
            mae = mean_absolute_error(test, forecast)
            
            # Prediksi harga di masa depan
            future_forecast = model_fit.forecast(steps=self.forecast_days)[-1]
            
            return future_forecast, {'mse': mse, 'mae': mae}
        except Exception as e:
            logger.error(f"ARIMA error: {e}")
            return self.data['Close'].iloc[-1], {'error': str(e)}

    def predict_with_lstm(self):
        """Prediksi menggunakan model LSTM"""
        try:
            # Siapkan data
            data = self.data[['Close']].values
            scaled_data = self.scaler.fit_transform(data)
            
            # Buat dataset
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            
            # Reshape untuk LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Bagi data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Bangun model LSTM
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Latih model
            model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0)
            
            # Prediksi testing
            predictions = model.predict(X_test)
            predictions = self.scaler.inverse_transform(predictions)
            y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Evaluasi
            mse = mean_squared_error(y_test_orig, predictions)
            mae = mean_absolute_error(y_test_orig, predictions)
            
            # Prediksi masa depan
            last_60_days = scaled_data[-60:]
            future_predictions = []
            
            for _ in range(self.forecast_days):
                x_input = last_60_days.reshape((1, 60, 1))
                pred = model.predict(x_input, verbose=0)
                future_predictions.append(pred[0])
                last_60_days = np.append(last_60_days[1:], pred)
            
            future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_price = future_predictions[-1][0]
            
            return future_price, {'mse': mse, 'mae': mae}
        except Exception as e:
            logger.error(f"LSTM error: {e}")
            return self.data['Close'].iloc[-1], {'error': str(e)}

    def predict_with_random_forest(self):
        """Prediksi menggunakan Random Forest dengan fitur teknikal"""
        try:
            # Siapkan fitur
            df = self.data.copy()
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Buat fitur lag
            for i in range(1, 6):
                df[f'Close_Lag_{i}'] = df['Close'].shift(i)
                df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
            
            # Hapus baris dengan nilai NaN
            df = df.dropna()
            
            # Pisahkan fitur dan target
            X = df[['MA20', 'MA50', 'MA100', 'RSI', 'MACD', 'MACD_Signal', 
                    'Volatility', 'Price_Change', 'Volume_Change',
                    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Volume_Lag_1']]
            y = df['Close']
            
            # Bagi data
            split = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # Model Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Prediksi testing
            predictions = model.predict(X_test)
            
            # Evaluasi
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            # Prediksi masa depan (menggunakan data terbaru)
            latest_data = X.iloc[-1:].copy()
            future_price = model.predict(latest_data)[0]
            
            return future_price, {'mse': mse, 'mae': mae}
        except Exception as e:
            logger.error(f"Random Forest error: {e}")
            return self.data['Close'].iloc[-1], {'error': str(e)}

    def calculate_rsi(self, series, window=14):
        """Menghitung Relative Strength Index (RSI)"""
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def calculate_macd(self, series, slow=26, fast=12, signal=9):
        """Menghitung Moving Average Convergence Divergence (MACD)"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    def get_technical_indicators(self):
        """Mengembalikan indikator teknikal terbaru"""
        if self.data.empty:
            return {}
        
        latest = self.data.iloc[-1]
        return {
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_Signal'],
            'ma20': latest['MA20'],
            'ma50': latest['MA50'],
            'ma100': latest['MA100'],
            'volatility': latest['Volatility']
        }
