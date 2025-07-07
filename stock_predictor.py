import yfinance as yf
import pandas as pd
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.fetch_data()
        
    def fetch_data(self):
        """Mengambil data historis saham"""
        try:
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period="2y", interval="1d")
            
            if hist.empty or len(hist) < 50:
                logger.warning(f"Data tidak cukup untuk {self.ticker}")
                return pd.DataFrame()
            
            # Hitung moving averages
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            
            # Hitung volatilitas
            hist['Volatility'] = hist['Close'].pct_change().rolling(window=14).std()
            
            # Hapus baris dengan nilai NaN
            hist = hist.dropna()
            
            return hist
        except Exception as e:
            logger.error(f"Error mengambil data {self.ticker}: {e}")
            return pd.DataFrame()
    
    def predict_trend(self):
        """Memprediksi trend dengan model sederhana"""
        if self.data.empty or len(self.data) < 50:
            return "Netral", self.data['Close'].iloc[-1] if not self.data.empty else 0
        
        try:
            # Gunakan moving average crossover sebagai prediktor
            last_price = self.data['Close'].iloc[-1]
            ma20 = self.data['MA20'].iloc[-1]
            ma50 = self.data['MA50'].iloc[-1]
            volatility = self.data['Volatility'].iloc[-1]

            if ma20 > ma50 and last_price > ma20:
                return "Naik", last_price * (1 + volatility)
            elif ma20 < ma50 and last_price < ma20:
                return "Turun", last_price * (1 - volatility)
            else:
                return "Netral", last_price
        except Exception as e:
            logger.error(f"Error memprediksi trend: {e}")
            return "Netral", self.data['Close'].iloc[-1] if not self.data.empty else 0

    def get_technical_indicators(self):
        """Mengembalikan indikator teknikal terbaru"""
        if self.data.empty:
            return {}
        
        try:
            latest = self.data.iloc[-1]
            return {
                'ma20': latest['MA20'],
                'ma50': latest['MA50'],
                'volatility': latest['Volatility'],
                'last_price': latest['Close']
            }
        except Exception as e:
            logger.error(f"Error mengambil indikator teknikal: {e}")
            return {}
