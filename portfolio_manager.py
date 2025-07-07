import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import logging
import concurrent.futures
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, portfolio_df: pd.DataFrame):
        self.df = portfolio_df.copy()
        self.lot_col = 'Lot Balance'
        self._validate_dataframe()
        
        # Convert numeric columns
        self.df[self.lot_col] = pd.to_numeric(self.df[self.lot_col], errors='coerce').fillna(0).astype(int)
        self.df['Avg Price'] = pd.to_numeric(self.df['Avg Price'], errors='coerce').fillna(0.0)
        
        # Initialize new columns
        self.df['Last Updated'] = pd.NaT
        self.df['Data Status'] = 'Pending'
        
        # Cache for stock data
        self.price_cache = {}
        self.cache_expiry = 300  # 5 minutes in seconds

    def _validate_dataframe(self):
        """Validate required columns exist in the dataframe"""
        required_columns = ['Ticker', self.lot_col, 'Avg Price']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataframe missing required columns: {', '.join(missing)}")

    def _fetch_price(self, ticker: str) -> float:
        """Fetch current price for a single ticker with caching and error handling"""
        current_time = datetime.now()
        
        # Check cache first
        if ticker in self.price_cache:
            price, timestamp = self.price_cache[ticker]
            if (current_time - timestamp).total_seconds() < self.cache_expiry:
                return price
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d", interval="1m")
            
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                self.price_cache[ticker] = (price, current_time)
                return price
                
            logger.warning(f"No data found for {ticker}")
            return np.nan
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return np.nan

    def update_realtime_prices(self, max_workers: int = 5):
        """Update prices using parallel processing with progress tracking"""
        tickers = self.df['Ticker'].unique().tolist()
        
        # Update status
        self.df['Data Status'] = 'Pending'
        self.df['Last Updated'] = pd.NaT
        
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            price_results = list(executor.map(self._fetch_price, tickers))
        
        # Create price mapping
        price_map = dict(zip(tickers, price_results))
        
        # Update dataframe
        current_prices = []
        statuses = []
        timestamps = []
        
        for idx, row in self.df.iterrows():
            ticker = row['Ticker']
            price = price_map.get(ticker, np.nan)
            
            # Handle missing prices
            if pd.isna(price):
                price = row['Avg Price']
                status = 'Error'
            else:
                status = 'Success'
                
            current_prices.append(price)
            statuses.append(status)
            timestamps.append(datetime.now())
        
        self.df['Current Price'] = current_prices
        self.df['Data Status'] = statuses
        self.df['Last Updated'] = timestamps
        
        # Calculate derived values
        self._calculate_portfolio_values()
        
    def _calculate_portfolio_values(self):
        """Calculate all derived portfolio metrics"""
        self.df['Current Value'] = self.df[self.lot_col] * self.df['Current Price']
        self.df['Total Investment'] = self.df[self.lot_col] * self.df['Avg Price']
        
        # Calculate profit/loss with zero-division protection
        invested = self.df[self.lot_col] * self.df['Avg Price']
        self.df['Profit/Loss'] = self.df['Current Value'] - invested
        
        # Calculate percentage profit/loss
        profit_pct = np.zeros(len(self.df))
        valid_mask = (invested > 0)
        profit_pct[valid_mask] = (
            (self.df['Current Value'][valid_mask] / invested[valid_mask] - 1
        ) * 100
        self.df['Profit/Loss %'] = profit_pct
        
        # Add position weight
        total_value = self.df['Current Value'].sum()
        if total_value > 0:
            self.df['Weight %'] = (self.df['Current Value'] / total_value) * 100
        else:
            self.df['Weight %'] = 0.0

    def analyze_dca(self):
        """Analyze dollar-cost averaging performance"""
        self.update_realtime_prices()
        return self.df

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get portfolio summary metrics with validation"""
        total_investment = self.df['Total Investment'].sum()
        total_current_value = self.df['Current Value'].sum()
        total_profit = total_current_value - total_investment
        
        if total_investment > 0:
            total_profit_percent = (total_current_value / total_investment - 1) * 100
        else:
            total_profit_percent = 0.0
            
        return {
            'total_investment': total_investment,
            'total_current_value': total_current_value,
            'total_profit': total_profit,
            'total_profit_percent': total_profit_percent,
            'positions_count': len(self.df),
            'success_rate': (self.df['Data Status'] == 'Success').mean() * 100
        }

    def get_dataframe(self) -> pd.DataFrame:
        """Get portfolio dataframe with proper column ordering"""
        cols_order = [
            'Ticker', self.lot_col, 'Avg Price', 'Current Price',
            'Total Investment', 'Current Value', 'Profit/Loss', 
            'Profit/Loss %', 'Weight %', 'Data Status', 'Last Updated'
        ]
        return self.df[[col for col in cols_order if col in self.df.columns]]
    
    def save_to_excel(self, filename: str):
        """Save portfolio to Excel file with formatting"""
        df = self.get_dataframe()
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Portfolio', index=False)
            
            # Apply formatting
            workbook = writer.book
            worksheet = writer.sheets['Portfolio']
            
            # Format for currency and percentages
            money_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # Apply formatting to appropriate columns
            money_cols = ['Avg Price', 'Current Price', 'Total Investment', 
                         'Current Value', 'Profit/Loss']
            percent_cols = ['Profit/Loss %', 'Weight %']
            
            for col in money_cols:
                if col in df.columns:
                    col_idx = df.columns.get_loc(col)
                    worksheet.set_column(col_idx, col_idx, None, money_format)
            
            for col in percent_cols:
                if col in df.columns:
                    col_idx = df.columns.get_loc(col)
                    worksheet.set_column(col_idx, col_idx, None, percent_format)
            
            # Autofit columns
            for idx, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(idx, idx, max_len)

    def add_position(self, ticker: str, lots: int, avg_price: float):
        """Add a new position to the portfolio"""
        new_row = {
            'Ticker': ticker,
            self.lot_col: lots,
            'Avg Price': avg_price,
            'Data Status': 'Pending',
            'Last Updated': pd.NaT
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.update_realtime_prices()

    def remove_position(self, ticker: str, all_lots: bool = True, lots_to_remove: Optional[int] = None):
        """Remove a position or reduce lots"""
        if ticker not in self.df['Ticker'].values:
            raise ValueError(f"Ticker {ticker} not found in portfolio")
            
        idx = self.df.index[self.df['Ticker'] == ticker].tolist()[0]
        
        if all_lots:
            self.df = self.df.drop(index=idx)
        else:
            if lots_to_remove is None:
                raise ValueError("Must specify lots_to_remove when all_lots=False")
                
            current_lots = self.df.at[idx, self.lot_col]
            if lots_to_remove > current_lots:
                raise ValueError("Cannot remove more lots than currently held")
                
            self.df.at[idx, self.lot_col] = current_lots - lots_to_remove
            
        self._calculate_portfolio_values()
