import yfinance as yf
import pandas as pd

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self.fetch_data()

    def fetch_data(self):
        try:
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period="1y")
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            return hist
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def predict_trend(self):
        if self.data.empty:
            return None

        last_price = self.data['Close'].iloc[-1]
        ma20 = self.data['MA20'].iloc[-1]
        ma50 = self.data['MA50'].iloc[-1]

        if ma20 > ma50 and last_price > ma20:
            return "Naik", last_price * 1.05
        elif ma20 < ma50 and last_price < ma20:
            return "Turun", last_price * 0.95
        else:
            return "Netral", last_price * 1.01

    def calculate_rsi(self, window=14):
        close = self.data['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, slow=26, fast=12, signal=9):
        close = self.data['Close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
