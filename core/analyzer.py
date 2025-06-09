# core/analyzer.py
from services.data_fetcher import get_price_data, get_stock_info
from services.fundamental_analysis import summarize_fundamental
from services.technical_analysis import (
    calculate_moving_average, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, generate_technical_signals
)
from services.projection_model import project_all
from services.recommender import give_recommendation

class SahamAnalyzer:
    def __init__(self, ticker, days):
        self.ticker = ticker
        self.days = days
        self.df = None
        self.info = None
        self.fundamental = None
        self.signals = None
        self.projections = None

    def fetch_data(self):
        self.df = get_price_data(self.ticker, days=self.days)
        self.info = get_stock_info(self.ticker)

    def process_fundamental(self):
        if self.info:
            self.fundamental = summarize_fundamental(self.info)

    def apply_technical_indicators(self, rsi=True, macd=True, bbands=False):
        if self.df is not None:
            self.df = calculate_moving_average(self.df, 20)
            self.df = calculate_moving_average(self.df, 50)
            if rsi:
                self.df = calculate_rsi(self.df)
            if macd:
                self.df = calculate_macd(self.df)
            if bbands:
                self.df = calculate_bollinger_bands(self.df)
            self.signals = generate_technical_signals(self.df)

    def make_projections(self, growth_rate=0.1, pe_target=15):
        eps = self.info.get("eps", 0)
        current_price = self.df['Close'].iloc[-1] if self.df is not None else 0
        self.projections = project_all(eps, current_price, growth_rate, pe_target)

    def get_recommendation(self):
        return give_recommendation(
            self.fundamental["Valuation"], 
            self.signals, 
            self.projections,
            self.info.get("dividend_yield")
        )
