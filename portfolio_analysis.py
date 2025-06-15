# portfolio_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
from scipy.stats import norm
from scipy.optimize import minimize

# ======================
# DATA MANAGEMENT MODULE
# ======================
class PortfolioManager:
    def __init__(self):
        self.df = self.load_portfolio()
        self.simulated_data = self.generate_historical_data()
        self.new_stocks = self.get_new_stocks()
        self.dividend_history = self.load_dividend_history()
        self.last_update = datetime.now()
        
    @staticmethod
    def load_portfolio():
        """Initialize portfolio data"""
        df = pd.DataFrame({
            'Stock': ['AADI', 'ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'LPPF', 'PGAS', 'PTBA', 'UNVR', 'WIIM'],
            'Ticker': ['AADI.JK', 'ADRO.JK', 'ANTM.JK', 'BFIN.JK', 'BJBR.JK', 'BSSR.JK', 'LPPF.JK', 'PGAS.JK', 'PTBA.JK', 'UNVR.JK', 'WIIM.JK'],
            'Lot Balance': [5.0, 17.0, 15.0, 30.0, 23.0, 11.0, 5.0, 10.0, 4.0, 60.0, 5.0],
            'Balance': [500, 1700, 1500, 3000, 2300, 1100, 500, 1000, 400, 6000, 500],
            'Avg Price': [7300, 2605, 1423, 1080, 1145, 4489, 1700, 1600, 2400, 1860, 871],
            'Stock Value': [3650000, 4428500, 2135000, 3240000, 2633500, 4938000, 850000, 1600000, 960000, 11162500, 435714],
            'Market Price': [7225.0, 2200.0, 3110.0, 905.0, 850.0, 4400.0, 1745.0, 1820.0, 2890.0, 1730.0, 835.0],
            'Unrealized': [-37500, -688500, 2530000, -525000, -678500, -98000, 22500, 220000, 196000, -782500, -18215],
            'Dividend Yield': [2.5, 3.0, 1.8, 2.0, 3.5, 2.8, 1.5, 4.0, 3.2, 2.7, 1.9]
        })
        
        df['Market Value'] = df['Balance'] * df['Market Price']
        return df
    
    def generate_historical_data(self):
        """Generate realistic historical price data"""
        np.random.seed(42)
        dates = pd.date_range(end='2025-05-31', periods=1000, freq='D')
        data = {}
        
        for stock in self.df['Stock']:
            base_price = self.df[self.df['Stock'] == stock]['Market Price'].iloc[0]
            volatility = base_price * 0.02
            
            prices = [base_price]
            for _ in range(1, len(dates)):
                change = np.random.normal(0, volatility)
                if stock in ['ANTM', 'PTBA', 'PGAS']:
                    change += volatility * 0.1
                prices.append(prices[-1] + change)
            
            data[stock] = pd.DataFrame({'Date': dates, 'Price': prices})
        return data
    
    @staticmethod
    def get_new_stocks():
        """Get new stock recommendations"""
        return pd.DataFrame({
            'Stock': ['TLKM', 'BBCA', 'BMRI', 'ASII'],
            'Ticker': ['TLKM.JK', 'BBCA.JK', 'BMRI.JK', 'ASII.JK'],
            'Sector': ['Telecom', 'Banking', 'Banking', 'Automotive'],
            'Dividend Yield': [4.5, 3.2, 3.8, 2.9],
            'Growth Rate': [8.0, 10.0, 9.5, 7.0],
            'Current Price': [3500, 9500, 6000, 4500],
            'Risk Level': ['Low', 'Medium', 'Medium', 'High']
        })
    
    @staticmethod
    def load_dividend_history():
        """Load dividend history data"""
        return pd.DataFrame({
            'Stock': ['AADI', 'ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'LPPF', 'PGAS', 'PTBA', 'UNVR', 'WIIM'],
            'Dividend_Date': ['2024-06-01', '2024-05-15', '2024-07-01', '2024-06-15', '2024-05-30', '2024-06-20', '2024-07-10', '2024-06-05', '2024-06-25', '2024-05-20', '2024-07-05'],
            'Dividend_Amount': [50.0, 75.0, 30.0, 25.0, 40.0, 60.0, 20.0, 80.0, 70.0, 50.0, 25.0]
        })
    
    def update_real_time_prices(self):
        """Fetch real-time market prices using Yahoo Finance API"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_tickers = list(self.df['Ticker']) + list(self.new_stocks['Ticker'])
            prices = {}
            
            for i, ticker in enumerate(all_tickers):
                status_text.text(f"Mengambil data untuk {ticker}...")
                progress_bar.progress((i + 1) / len(all_tickers))
                
                try:
                    stock_data = yf.Ticker(ticker)
                    hist = stock_data.history(period='1d', auto_adjust=True)
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prices[ticker] = float(current_price)
                    else:
                        if ticker in self.df['Ticker'].values:
                            prices[ticker] = float(self.df[self.df['Ticker'] == ticker]['Market Price'].iloc[0])
                        elif ticker in self.new_stocks['Ticker'].values:
                            prices[ticker] = float(self.new_stocks[self.new_stocks['Ticker'] == ticker]['Current Price'].iloc[0])
                except Exception as e:
                    st.warning(f"Error mengambil data untuk {ticker}: {str(e)}")
                    if ticker in self.df['Ticker'].values:
                        prices[ticker] = float(self.df[self.df['Ticker'] == ticker]['Market Price'].iloc[0])
                    elif ticker in self.new_stocks['Ticker'].values:
                        prices[ticker] = float(self.new_stocks[self.new_stocks['Ticker'] == ticker]['Current Price'].iloc[0])
            
            for idx, row in self.df.iterrows():
                ticker = row['Ticker']
                if ticker in prices:
                    self.df.at[idx, 'Market Price'] = prices[ticker]
            
            for idx, row in self.new_stocks.iterrows():
                ticker = row['Ticker']
                if ticker in prices:
                    self.new_stocks.at[idx, 'Current Price'] = prices[ticker]
            
            self.df['Market Value'] = self.df['Balance'] * self.df['Market Price']
            self.df['Unrealized'] = self.df['Market Value'] - self.df['Stock Value']
            
            self.last_update = datetime.now()
            return True
        except Exception as e:
            st.error(f"Error memperbarui harga: {str(e)}")
            return False
        finally:
            progress_bar.empty()
            status_text.empty()

# ===================
# ANALYSIS MODULE
# ===================
class PortfolioAnalyzer:
    def __init__(self, portfolio_manager):
        self.pm = portfolio_manager
        
    def portfolio_summary(self):
        """Calculate key portfolio metrics"""
        df = self.pm.df
        total_dividend = self.calculate_total_dividend()
        return {
            'total_invested': df['Stock Value'].sum(),
            'total_market_value': df['Market Value'].sum(),
            'total_unrealized': df['Unrealized'].sum(),
            'return_pct': (df['Unrealized'].sum() / df['Stock Value'].sum()) * 100,
            'total_dividend': total_dividend
        }
    
    def calculate_total_dividend(self):
        """Calculate total dividend income"""
        total_dividend = 0
        for _, row in self.pm.df.iterrows():
            stock = row['Stock']
            balance = row['Balance']
            div_data = self.pm.dividend_history[self.pm.dividend_history['Stock'] == stock]
            if not div_data.empty:
                total_dividend += div_data['Dividend_Amount'].sum() * balance
        return total_dividend
    
    def predict_price(self, stock, days=30):
        """Predict future prices using ML model"""
        if stock not in self.pm.simulated_data:
            return None, None, None
            
        data = self.pm.simulated_data[stock].copy()
        data['Days'] = (data['Date'] - data['Date'].min()).dt.days
        
        data['MA7'] = data['Price'].rolling(window=7).mean()
        data['MA30'] = data['Price'].rolling(window=30).mean()
        data = data.dropna()
        
        X = data[['Days', 'MA7', 'MA30']]
        y = data['Price']
        
        model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
        model.fit(X, y)
        
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
        future_days = [(fd - data['Date'].min()).days for fd in future_dates]
        
        last_ma7 = data['MA7'].iloc[-1]
        last_ma30 = data['MA30'].iloc[-1]
        future_ma7 = []
        future_ma30 = []
        
        for i in range(days):
            if i < 7:
                future_ma7.append(np.mean(data['Price'].iloc[-(7-i):]))
            else:
                future_ma7.append(last_ma7)
                
            if i < 30:
                future_ma30.append(np.mean(data['Price'].iloc[-(30-i):]))
            else:
                future_ma30.append(last_ma30)
        
        future_X = pd.DataFrame({
            'Days': future_days,
            'MA7': future_ma7,
            'MA30': future_ma30
        })
        predictions = model.predict(future_X)
        
        return future_dates, predictions, predictions[-1]
    
    def what_if_simulation(self, stock, price_change_pct):
        """Simulate portfolio impact of price changes"""
        sim_df = self.pm.df.copy()
        idx = sim_df[sim_df['Stock'] == stock].index
        
        if not idx.empty:
            original_price = sim_df.loc[idx, 'Market Price'].values[0]
            new_price = original_price * (1 + price_change_pct / 100)
            sim_df.loc[idx, 'Market Price'] = new_price
            sim_df['Market Value'] = sim_df['Balance'] * sim_df['Market Price']
            sim_df['Unrealized'] = sim_df['Market Value'] - sim_df['Stock Value']
            
            return {
                'new_total_market': sim_df['Market Value'].sum(),
                'new_unrealized': sim_df['Unrealized'].sum(),
                'sim_df': sim_df
            }
        return None
    
    def generate_recommendations(self):
        """Generate buy/sell recommendations"""
        recommendations = []
        
        for _, row in self.pm.df.iterrows():
            stock = row['Stock']
            unrealized_pct = (row['Unrealized'] / row['Stock Value']) * 100
            dividend_yield = row['Dividend Yield']
            
            trend = 0
            if stock in self.pm.simulated_data:
                data = self.pm.simulated_data[stock]
                if len(data) > 10:
                    trend = (data['Price'].iloc[-1] / data['Price'].iloc[-10] - 1) * 100
            
            if unrealized_pct < -15 or trend < -5 or dividend_yield < 2.0:
                rec = 'Sell'
                reason = 'Kerugian besar, tren menurun, atau dividen rendah'
                urgency = 'High'
            elif unrealized_pct > 20 or trend > 8 or dividend_yield > 4.0:
                rec = 'Buy More'
                reason = 'Kinerja kuat, tren naik, atau dividen tinggi'
                urgency = 'Medium'
            elif unrealized_pct > 5 or trend > 3:
                rec = 'Hold/Buy'
                reason = 'Kinerja positif'
                urgency = 'Low'
            elif unrealized_pct < -5:
                rec = 'Hold/Sell'
                reason = 'Kinerja di bawah rata-rata'
                urgency = 'Monitor'
            else:
                rec = 'Hold'
                reason = 'Kinerja stabil'
                urgency = 'Low'
                
            recommendations.append({
                'Stock': stock,
                'Recommendation': rec,
                'Reason': reason,
                'Urgency': urgency,
                'Unrealized %': f"{unrealized_pct:.1f}%",
                '30d Trend %': f"{trend:.1f}%",
                'Dividend Yield %': f"{dividend_yield:.1f}%"
            })
            
        return pd.DataFrame(recommendations)
    
    def optimize_portfolio(self, risk_free_rate=0.05):
        """Optimize portfolio allocation using Markowitz model"""
        returns_data = {}
        for stock, data in self.pm.simulated_data.items():
            df = data.copy()
            df['Return'] = df['Price'].pct_change().dropna()
            returns_data[stock] = df['Return']
        
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return None
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        num_assets = len(returns_df.columns)
        weights = np.ones(num_assets) / num_assets
        
        def portfolio_performance(weights, returns, cov_matrix, risk_free_rate):
            portfolio_return = np.sum(returns * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return portfolio_return, portfolio_volatility, sharpe_ratio
        
        def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
            return -portfolio_performance(weights, returns, cov_matrix, risk_free_rate)[2]
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x})  # Weights >= 0
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(negative_sharpe_ratio, weights,
                        args=(expected_returns, cov_matrix, risk_free_rate),
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            opt_return, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, expected_returns, cov_matrix, risk_free_rate)
            return {
                'weights': dict(zip(returns_df.columns, optimal_weights)),
                'return': opt_return,
                'volatility': opt_vol,
                'sharpe_ratio': opt_sharpe
            }
        return None

# ===================
# VISUALIZATION MODULE
# ===================
class PortfolioVisualizer:
    @staticmethod
    def portfolio_pie(df):
        """Create portfolio composition pie chart"""
        fig = px.pie(df, values='Market Value', names='Stock', 
                     title='Komposisi Portofolio', hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    @staticmethod
    def performance_bar(df):
        """Create unrealized gain/loss bar chart"""
        df = df.copy()
        df['Color'] = df['Unrealized'].apply(lambda x: 'green' if x >= 0 else 'red')
        fig = px.bar(df, x='Stock', y='Unrealized', color='Color',
                     title='Keuntungan/Rugi Belum Direalisasi per Saham',
                     labels={'Unrealized': 'Keuntungan/Rugi (Rp)'})
        fig.update_layout(showlegend=False)
        return fig
    
    @staticmethod
    def price_prediction_plot(history, forecast, stock):
        """Plot price prediction with confidence interval"""
        history = history.rename(columns={'Price': 'Value'})
        history['Type'] = 'Historical'
        
        forecast_df = pd.DataFrame({
            'Date': forecast['dates'],
            'Value': forecast['predictions'],
            'Type': 'Forecast'
        })
        
        combined = pd.concat([history, forecast_df])
        
        fig = px.line(combined, x='Date', y='Value', color='Type',
                      title=f"Prediksi Harga untuk {stock}",
                      labels={'Value': 'Harga (Rp)'})
        
        if 'upper' in forecast and 'lower' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                y=forecast['upper'] + forecast['lower'][::-1],
                fill='toself',
                fillcolor='rgba(100, 150, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Interval Keyakinan'
            ))
        
        return fig
    
    @staticmethod
    def dividend_bar(dividend_history, df):
        """Create dividend income bar chart"""
        div_data = dividend_history.merge(df[['Stock', 'Balance']], on='Stock')
        div_data['Total_Dividend'] = div_data['Dividend_Amount'] * div_data['Balance']
        
        fig = px.bar(div_data, x='Stock', y='Total_Dividend',
                     title='Pendapatan Dividen per Saham',
                     labels={'Total_Dividend': 'Pendapatan Dividen (Rp)'},
                     color='Total_Dividend',
                     color_continuous_scale='Blues')
        return fig
    
    @staticmethod
    def allocation_plot(optimal_weights, current_weights):
        """Create comparison of current vs optimal portfolio allocation"""
        weights_df = pd.DataFrame({
            'Stock': list(optimal_weights.keys()),
            'Current': list(current_weights.values()),
            'Optimal': list(optimal_weights.values())
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Alokasi Saat Ini', x=weights_df['Stock'], y=weights_df['Current']),
            go.Bar(name='Alokasi Optimal', x=weights_df['Stock'], y=weights_df['Optimal'])
        ])
        fig.update_layout(
            title='Perbandingan Alokasi Portofolio',
            yaxis_title='Bobot (%)',
            barmode='group'
        )
        return fig

# ===================
# RISK ANALYSIS MODULE
# ===================
class RiskAnalyzer:
    def __init__(self, portfolio_manager):
        self.pm = portfolio_manager
        self.portfolio_returns = self.calculate_portfolio_returns()
        
    def calculate_stock_returns(self):
        """Calculate daily returns for each stock from historical data"""
        returns_data = {}
        for stock, data in self.pm.simulated_data.items():
            df = data.copy()
            df['Return'] = df['Price'].pct_change().dropna()
            returns_data[stock] = df
        return returns_data
    
    def calculate_portfolio_returns(self):
        """Calculate historical portfolio returns based on current weights"""
        returns_data = self.calculate_stock_returns()
        
        returns_df = pd.DataFrame()
        for stock in self.pm.df['Stock']:
            if stock in returns_data:
                returns_df[stock] = returns_data[stock]['Return']
        
        if returns_df.empty:
            return None
        
        total_value = self.pm.df['Market Value'].sum()
        weights = self.pm.df.set_index('Stock')['Market Value'] / total_value
        weights = weights[returns_df.columns].values
        portfolio_returns = returns_df.dot(weights)
        return portfolio_returns.dropna()
    
    def portfolio_volatility(self, annualize=True):
        """Calculate portfolio volatility"""
        if self.portfolio_returns is None:
            return 0
        vol = self.portfolio_returns.std()
        if annualize:
            vol = vol * np.sqrt(252)
        return vol
    
    def value_at_risk(self, confidence_level=0.95, method='historical'):
        """Calculate Value at Risk (VaR)"""
        if self.portfolio_returns is None:
            return 0
        if method == 'historical':
            return -np.percentile(self.portfolio_returns, 100*(1-confidence_level))
        else:
            mean = self.portfolio_returns.mean()
            std = self.portfolio_returns.std()
            z_score = norm.ppf(1-confidence_level)
            return -(mean + z_score * std)
    
    def expected_shortfall(self, confidence_level=0.95):
        """Calculate Expected Shortfall (ES)"""
        if self.portfolio_returns is None:
            return 0
        var = self.value_at_risk(confidence_level, 'historical')
        losses = -self.portfolio_returns
        return losses[losses > var].mean()
    
    def beta_analysis(self):
        """Calculate beta coefficients relative to market index"""
        try:
            market = yf.download('^JKSE', period='1y', auto_adjust=True)['Close'].pct_change().dropna()
            beta_values = {}
            for stock in self.pm.df['Stock']:
                if stock in self.pm.simulated_data:
                    stock_returns = self.pm.simulated_data[stock]['Price'].pct_change().dropna()
                    common_dates = stock_returns.index.intersection(market.index)
                    
                    if len(common_dates) > 10:
                        cov_matrix = np.cov(
                            stock_returns.loc[common_dates], 
                            market.loc[common_dates]
                        )
                        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                        beta_values[stock] = beta
            return beta_values
        except Exception as e:
            print(f"Error in beta calculation: {e}")
            return {}
    
    def stress_test(self, scenario='crisis'):
        """Stress test portfolio under different market conditions"""
        scenarios = {
            'crisis': -0.30,
            'recession': -0.15,
            'bull': 0.20,
            'correction': -0.10
        }
        
        if scenario not in scenarios:
            return None
        
        betas = self.beta_analysis()
        scenario_return = scenarios[scenario]
        portfolio_impact = 0
        stock_impacts = {}
        
        for idx, row in self.pm.df.iterrows():
            stock = row['Stock']
            beta = betas.get(stock, 1.0)
            expected_return = beta * scenario_return
            current_value = row['Market Value']
            impact = current_value * expected_return
            portfolio_impact += impact
            stock_impacts[stock] = {
                'Beta': beta,
                'Expected Return': expected_return,
                'Impact': impact
            }
        
        return {
            'scenario': scenario,
            'portfolio_impact': portfolio_impact,
            'stock_impacts': stock_impacts
        }
    
    def diversification_metrics(self):
        """Calculate diversification metrics"""
        returns_data = self.calculate_stock_returns()
        returns_df = pd.DataFrame()
        for stock, data in returns_data.items():
            returns_df[stock] = data['Return']
        
        portfolio_stocks = set(self.pm.df['Stock'])
        available_stocks = set(returns_df.columns)
        common_stocks = list(portfolio_stocks & available_stocks)
        
        if not common_stocks:
            return {
                'correlation_matrix': pd.DataFrame(),
                'average_correlation': 0,
                'diversification_ratio': 0
            }
        
        returns_df = returns_df[common_stocks]
        corr_matrix = returns_df.corr()
        
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        avg_correlation = corr_matrix.where(mask).mean().mean()
        
        filtered_df = self.pm.df[self.pm.df['Stock'].isin(common_stocks)]
        
        weighted_vol = filtered_df['Market Value'] / filtered_df['Market Value'].sum()
        weighted_vol = weighted_vol.values
        individual_vol = returns_df.std().values
        portfolio_vol = self.portfolio_volatility(annualize=False)
        
        individual_vol = np.nan_to_num(individual_vol, nan=0.0)
        
        diversification_ratio = np.sum(weighted_vol * individual_vol) / portfolio_vol
        
        return {
            'correlation_matrix': corr_matrix,
            'average_correlation': avg_correlation,
            'diversification_ratio': diversification_ratio
        }

# ===================
# STREAMLIT APP
# ===================
def main():
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = PortfolioManager()
    
    pm = st.session_state.portfolio
    analyzer = PortfolioAnalyzer(pm)
    visualizer = PortfolioVisualizer()
    
    st.title("📊 Dashboard Analisis Portofolio Lanjutan")
    st.caption("Alat interaktif untuk manajemen portofolio dan analisis saham")
    
    st.header("📈 Data Pasar Real-time")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Perbarui Harga Pasar", type="primary", help="Ambil harga pasar terbaru dari Yahoo Finance"):
            if pm.update_real_time_prices():
                st.success("Harga pasar berhasil diperbarui!")
                st.session_state.portfolio = pm
                st.rerun()
    
    with col2:
        update_time = pm.last_update.strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Pembaruan terakhir: {update_time}")
        st.progress(100, text="Siap untuk pembaruan")
    
    st.header("📊 Ringkasan Portofolio")
    summary = analyzer.portfolio_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Investasi", f"Rp {summary['total_invested']:,.0f}")
    col2.metric("Nilai Saat Ini", f"Rp {summary['total_market_value']:,.0f}", 
                f"{summary['return_pct']:.2f}%")
    col3.metric("Keuntungan/Rugi Belum Direalisasi", f"Rp {summary['total_unrealized']:,.0f}", 
                f"{summary['return_pct']:.2f}%", delta_color="inverse")
    col4.metric("Total Pendapatan Dividen", f"Rp {summary['total_dividend']:,.0f}")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(visualizer.portfolio_pie(pm.df), use_container_width=True)
    with col2:
        st.plotly_chart(visualizer.performance_bar(pm.df), use_container_width=True)
    
    st.header("📋 Detail Saham Real-time")
    
    pm.df['Unrealized %'] = (pm.df['Unrealized'] / pm.df['Stock Value']) * 100
    pm.df['Daily Change'] = (pm.df['Market Price'] / pm.df['Avg Price'] - 1) * 100
    pm.df['Current Value'] = pm.df['Balance'] * pm.df['Market Price']
    
    formatted_df = pm.df[['Stock', 'Balance', 'Avg Price', 'Market Price', 
                         'Daily Change', 'Current Value', 'Unrealized', 'Unrealized %', 'Dividend Yield']].copy()
    
    formatted_df['Avg Price'] = formatted_df['Avg Price'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Market Price'] = formatted_df['Market Price'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Current Value'] = formatted_df['Current Value'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Unrealized'] = formatted_df['Unrealized'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Daily Change'] = formatted_df['Daily Change'].apply(lambda x: f"{x:.2f}%")
    formatted_df['Unrealized %'] = formatted_df['Unrealized %'].apply(lambda x: f"{x:.2f}%")
    formatted_df['Dividend Yield'] = formatted_df['Dividend Yield'].apply(lambda x: f"{x:.2f}%")
    
    def color_negative_red(val):
        if isinstance(val, str) and '%' in val:
            num_val = float(val.replace('%', ''))
            color = 'red' if num_val < 0 else 'green'
        elif isinstance(val, str) and 'Rp' in val:
            num_val = float(val.replace('Rp', '').replace(',', '').strip())
            color = 'red' if num_val < 0 else 'green'
        else:
            return ''
        return f'color: {color}'
    
    styled_df = formatted_df.style.map(color_negative_red, 
                                     subset=['Daily Change', 'Unrealized', 'Unrealized %'])
    
    st.dataframe(styled_df, height=400, use_container_width=True)
    
    st.header("🔮 Prediksi Harga AI")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_stock = st.selectbox("Pilih Saham", pm.df['Stock'])
        days = st.slider("Periode Prediksi (hari)", 7, 90, 30)
        
    with col2:
        if selected_stock:
            with st.spinner("Membuat prediksi..."):
                dates, predictions, last_pred = analyzer.predict_price(selected_stock, days)
                
                if predictions is not None:
                    current_price = pm.df[pm.df['Stock'] == selected_stock]['Market Price'].iloc[0]
                    change_pct = (last_pred / current_price - 1) * 100
                    
                    st.metric(f"Harga Prediksi dalam {days} hari", 
                             f"Rp {last_pred:,.0f}",
                             f"{change_pct:.1f}%")
                    
                    volatility = current_price * 0.15 * np.sqrt(days/365)
                    upper = [p + volatility * (i/len(predictions)) for i, p in enumerate(predictions)]
                    lower = [p - volatility * (i/len(predictions)) for i, p in enumerate(predictions)]
                    
                    forecast_data = {
                        'dates': dates,
                        'predictions': predictions,
                        'upper': upper,
                        'lower': lower
                    }
                    
                    history = pm.simulated_data[selected_stock].tail(60)
                    st.plotly_chart(visualizer.price_prediction_plot(history, forecast_data, selected_stock),
                                  use_container_width=True)
    
    st.header("🔮 Analisis Skenario What If")
    col1, col2 = st.columns(2)
    
    with col1:
        sim_stock = st.selectbox("Pilih Saham", pm.df['Stock'], key='sim_stock')
        price_change = st.slider("Perubahan Harga (%)", -50.0, 50.0, 10.0, key='price_slider')
    
    with col2:
        if sim_stock:
            result = analyzer.what_if_simulation(sim_stock, price_change)
            if result:
                current_value = pm.df['Market Value'].sum()
                new_value = result['new_total_market']
                change = (new_value - current_value) / current_value * 100
                
                st.metric("Dampak Nilai Portofolio", 
                         f"Rp {new_value:,.0f}",
                         f"{change:.2f}%")
                
                st.metric("Dampak Keuntungan/Rugi Belum Direalisasi", 
                         f"Rp {result['new_unrealized']:,.0f}",
                         f"{(result['new_unrealized'] - summary['total_unrealized'])/summary['total_invested']*100:.2f}%")
    
    st.header("💡 Rekomendasi Beli/Jual")
    rec_df = analyzer.generate_recommendations()
    
    rec_colors = {
        'Sell': 'red',
        'Buy More': 'green',
        'Hold/Buy': 'lightgreen',
        'Hold/Sell': 'orange',
        'Hold': 'gray'
    }
    
    styled_rec = rec_df.style.apply(lambda x: [
        f"background-color: {rec_colors.get(v, 'white')}" for v in x
    ], subset=['Recommendation'])
    
    st.dataframe(styled_rec, height=400)
    
    st.header("💰 Pelacakan Dividen")
    st.subheader("Riwayat Dividen")
    div_df = pm.dividend_history.copy()
    div_df['Dividend_Date'] = pd.to_datetime(div_df['Dividend_Date']).dt.strftime('%Y-%m-%d')
    div_df['Dividend_Amount'] = div_df['Dividend_Amount'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(div_df, height=300)
    
    st.plotly_chart(visualizer.dividend_bar(pm.dividend_history, pm.df), use_container_width=True)
    
    st.header("⚖️ Optimasi Alokasi Portofolio")
    if st.button("Lakukan Optimasi Portofolio"):
        with st.spinner("Mengoptimalkan alokasi portofolio..."):
            opt_result = analyzer.optimize_portfolio()
            if opt_result:
                current_weights = pm.df['Market Value'] / pm.df['Market Value'].sum()
                current_weights = current_weights.to_dict()
                
                st.markdown("### Hasil Optimasi")
                col1, col2, col3 = st.columns(3)
                col1.metric("Pengembalian Tahunan", f"{opt_result['return']*100:.2f}%")
                col2.metric("Volatilitas Tahunan", f"{opt_result['volatility']*100:.2f}%")
                col3.metric("Rasio Sharpe", f"{opt_result['sharpe_ratio']:.2f}")
                
                weights_df = pd.DataFrame({
                    'Saham': list(opt_result['weights'].keys()),
                    'Bobot Optimal (%)': [w*100 for w in opt_result['weights'].values()]
                })
                weights_df['Bobot Optimal (%)'] = weights_df['Bobot Optimal (%)'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(weights_df, height=300)
                
                st.plotly_chart(visualizer.allocation_plot(opt_result['weights'], current_weights),
                              use_container_width=True)
            else:
                st.error("Gagal melakukan optimasi portofolio. Data tidak cukup.")
    
    st.header("📋 Manajemen Portofolio")
    
    # PERUBAHAN: Menu tambah saham baru menjadi inputan user
    st.subheader("Tambahkan Saham Baru")
    
    with st.expander("Form Tambah Saham Baru", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            stock_name = st.text_input("Nama Saham", placeholder="Contoh: BBCA")
            ticker = st.text_input("Kode Ticker", placeholder="Contoh: BBCA.JK")
            sector = st.selectbox("Sektor", ["Keuangan", "Teknologi", "Konsumsi", "Energi", "Properti", "Kesehatan", "Industri", "Infrastruktur", "Lainnya"])
            risk_level = st.select_slider("Tingkat Risiko", options=["Rendah", "Sedang", "Tinggi"])
        
        with col2:
            dividend_yield = st.number_input("Dividen Yield (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
            growth_rate = st.number_input("Pertumbuhan Tahunan (%)", min_value=-100.0, max_value=100.0, value=5.0, step=0.1)
            current_price = st.number_input("Harga Saat Ini (Rp)", min_value=1.0, value=1000.0, step=100.0)
            shares = st.number_input("Jumlah Saham", min_value=1, value=100, step=100)
    
        if st.button("Tambahkan ke Portofolio", key='add_stock'):
            if not stock_name or not ticker:
                st.error("Nama Saham dan Kode Ticker wajib diisi!")
            else:
                new_row = pd.DataFrame({
                    'Stock': [stock_name],
                    'Ticker': [ticker],
                    'Lot Balance': [shares / 100],
                    'Balance': [shares],
                    'Avg Price': [current_price],
                    'Stock Value': [shares * current_price],
                    'Market Price': [current_price],
                    'Market Value': [shares * current_price],
                    'Unrealized': [0],
                    'Dividend Yield': [dividend_yield]
                })
                
                pm.df = pd.concat([pm.df, new_row], ignore_index=True)
                st.success(f"Berhasil menambahkan {shares} saham {stock_name} ({ticker}) ke portofolio!")
                st.session_state.portfolio = pm
    
    st.subheader("Modifikasi Kepemilikan")
    mod_stock = st.selectbox("Pilih Saham", pm.df['Stock'], key='mod_stock')
    
    if mod_stock:
        current_balance = pm.df[pm.df['Stock'] == mod_stock]['Balance'].iloc[0]
        new_balance = st.number_input("Jumlah Saham Baru", min_value=0, value=int(current_balance))
        
        if st.button("Perbarui Saham"):
            idx = pm.df[pm.df['Stock'] == mod_stock].index
            if not idx.empty:
                row = pm.df.loc[idx]
                avg_price = row['Avg Price'].values[0]
                market_price = row['Market Price'].values[0]
                
                pm.df.loc[idx, 'Balance'] = new_balance
                pm.df.loc[idx, 'Lot Balance'] = new_balance / 100
                pm.df.loc[idx, 'Stock Value'] = new_balance * avg_price
                pm.df.loc[idx, 'Market Value'] = new_balance * market_price
                pm.df.loc[idx, 'Unrealized'] = pm.df.loc[idx, 'Market Value'] - pm.df.loc[idx, 'Stock Value']
                
                st.success(f"Berhasil memperbarui {mod_stock} menjadi {new_balance} saham!")
                st.session_state.portfolio = pm
    
    if st.button("Hapus Saham Terpilih", type="primary"):
        if mod_stock:
            pm.df = pm.df[pm.df['Stock'] != mod_stock]
            st.success(f"Berhasil menghapus {mod_stock} dari portofolio!")
            st.session_state.portfolio = pm
    
    st.subheader("Portofolio Saat Ini")
    st.dataframe(pm.df[['Stock', 'Balance', 'Avg Price', 'Market Price', 'Unrealized', 'Dividend Yield']], 
                 height=300)

    st.header("📊 Analisis Risiko")
    risk_analyzer = RiskAnalyzer(pm)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Volatilitas Portofolio (Tahunan)", f"{risk_analyzer.portfolio_volatility()*100:.2f}%")
    col2.metric("95% VaR (1-hari)", f"Rp {risk_analyzer.value_at_risk(0.95)*risk_analyzer.pm.df['Market Value'].sum():,.0f}")
    col3.metric("95% Expected Shortfall (1-hari)", f"Rp {risk_analyzer.expected_shortfall(0.95)*risk_analyzer.pm.df['Market Value'].sum():,.0f}")
    
    st.subheader("Eksposur Beta")
    betas = risk_analyzer.beta_analysis()
    if betas:
        beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta']).reset_index()
        beta_df.columns = ['Saham', 'Beta']
        fig = px.bar(beta_df, x='Saham', y='Beta', title='Beta Saham Relatif terhadap Indeks Pasar',
                     color='Beta', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak dapat menghitung nilai beta. Data pasar tidak tersedia.")
    
    st.subheader("Uji Stres")
    scenario = st.selectbox("Pilih Skenario", ['crisis', 'recession', 'correction', 'bull'])
    
    if st.button("Jalankan Uji Stres", key='stress_test'):
        results = risk_analyzer.stress_test(scenario)
        if results:
            st.markdown(f"### Hasil Skenario {scenario.capitalize()}")
            col1, col2 = st.columns(2)
            portfolio_value = pm.df['Market Value'].sum()
            impact_pct = (results['portfolio_impact'] / portfolio_value) * 100
            
            col1.metric("Dampak Nilai Portofolio", 
                       f"Rp {portfolio_value + results['portfolio_impact']:,.0f}",
                       f"{impact_pct:.2f}%")
            col2.metric("Perkiraan Kerugian/Keuntungan", 
                       f"Rp {results['portfolio_impact']:,.0f}")
            
            impact_df = pd.DataFrame.from_dict(results['stock_impacts'], orient='index')
            impact_df = impact_df.reset_index().rename(columns={'index': 'Saham'})
            impact_df['Dampak %'] = (impact_df['Impact'] / pm.df.set_index('Stock')['Market Value']) * 100
            
            fig = px.bar(impact_df, x='Saham', y='Dampak %', 
                         title='Dampak per Saham',
                         color='Dampak %', 
                         color_continuous_scale='RdYlGn' if scenario == 'bull' else 'RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Metrik Diversifikasi")
    div_metrics = risk_analyzer.diversification_metrics()
    
    col1, col2 = st.columns(2)
    col1.metric("Korelasi Rata-rata", f"{div_metrics['average_correlation']:.4f}")
    col2.metric("Rasio Diversifikasi", f"{div_metrics['diversification_ratio']:.2f}")
    
    st.markdown("**Matriks Korelasi**")
    fig = px.imshow(div_metrics['correlation_matrix'], 
                   text_auto=".2f", 
                   color_continuous_scale='RdYlGn',
                   title='Matriks Korelasi Saham')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
