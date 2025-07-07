import pandas as pd
import yfinance as yf
import numpy as np

class PortfolioManager:
    def __init__(self, portfolio_df):
        self.df = portfolio_df.copy()
        self.lot_col = 'Lot Balance'

    def update_realtime_prices(self):
        current_prices = []
        for idx, row in self.df.iterrows():
            ticker = row['Ticker']
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d", interval="5m")
                last_price = hist['Close'].iloc[-1] if not hist.empty else row['Avg Price']
            except Exception:
                last_price = row['Avg Price']
            current_prices.append(last_price)

        self.df['Current Price'] = current_prices
        self.df['Current Value'] = self.df[self.lot_col] * self.df['Current Price']
        self.df['Profit/Loss'] = self.df['Current Value'] - (self.df[self.lot_col] * self.df['Avg Price'])
        self.df['Profit/Loss %'] = (
            self.df['Current Value'] / (self.df[self.lot_col] * self.df['Avg Price']) - 1
        ) * 100

    def analyze_dca(self):
        self.update_realtime_prices()
        self.df['Total Investment'] = self.df[self.lot_col] * self.df['Avg Price']
        return self.df

    def get_summary_metrics(self):
        total_investment = self.df['Total Investment'].sum()
        total_current_value = self.df['Current Value'].sum()
        total_profit = total_current_value - total_investment
        total_profit_percent = (total_current_value / total_investment - 1) * 100
        return {
            'total_investment': total_investment,
            'total_current_value': total_current_value,
            'total_profit': total_profit,
            'total_profit_percent': total_profit_percent
        }

    def get_dataframe(self):
        return self.df
