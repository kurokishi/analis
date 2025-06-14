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
from scipy.stats import norm  # Ditambahkan untuk modul analisis risiko

# ======================
# DATA MANAGEMENT MODULE
# ======================
class PortfolioManager:
    def __init__(self):
        self.df = self.load_portfolio()
        self.simulated_data = self.generate_historical_data()
        self.new_stocks = self.get_new_stocks()
        self.last_update = datetime.now()
        
    @staticmethod
    def load_portfolio():
        """Initialize portfolio data"""
        return pd.DataFrame({
            'Stock': ['AADI', 'ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'LPPF', 'PGAS', 'PTBA', 'UNVR', 'WIIM'],
            'Ticker': ['AADI.JK', 'ADRO.JK', 'ANTM.JK', 'BFIN.JK', 'BJBR.JK', 'BSSR.JK', 'LPPF.JK', 'PGAS.JK', 'PTBA.JK', 'UNVR.JK', 'WIIM.JK'],
            'Lot Balance': [5.0, 17.0, 15.0, 30.0, 23.0, 11.0, 5.0, 10.0, 4.0, 60.0, 5.0],
            'Balance': [500, 1700, 1500, 3000, 2300, 1100, 500, 1000, 400, 6000, 500],
            'Avg Price': [7300, 2605, 1423, 1080, 1145, 4489, 1700, 1600, 2400, 1860, 871],
            'Stock Value': [3650000, 4428500, 2135000, 3240000, 2633500, 4938000, 850000, 1600000, 960000, 11162500, 435714],
            'Market Price': [7225, 2200, 3110, 905, 850, 4400, 1745, 1820, 2890, 1730, 835],
            'Unrealized': [-37500, -688500, 2530000, -525000, -678500, -98000, 22500, 220000, 196000, -782500, -18215]
        })
    
    def generate_historical_data(self):
        """Generate realistic historical price data"""
        np.random.seed(42)
        dates = pd.date_range(end='2025-05-31', periods=1000, freq='D')
        data = {}
        
        for stock in self.df['Stock']:
            base_price = self.df[self.df['Stock'] == stock]['Market Price'].iloc[0]
            volatility = base_price * 0.02  # 2% daily volatility
            
            # Create more realistic price series with trends
            prices = [base_price]
            for _ in range(1, len(dates)):
                change = np.random.normal(0, volatility)
                # Add slight upward bias for some stocks
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
    
    def update_real_time_prices(self):
        """Fetch real-time market prices using Yahoo Finance API"""
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get unique tickers
            all_tickers = list(self.df['Ticker']) + list(self.new_stocks['Ticker'])
            
            # Create a dictionary to store prices
            prices = {}
            
            # Fetch prices
            for i, ticker in enumerate(all_tickers):
                status_text.text(f"Mengambil data untuk {ticker}...")
                progress_bar.progress((i + 1) / len(all_tickers))
                
                try:
                    stock_data = yf.Ticker(ticker)
                    hist = stock_data.history(period='1d')
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prices[ticker] = current_price
                    else:
                        # Fallback to existing price if no data
                        if ticker in self.df['Ticker'].values:
                            prices[ticker] = self.df[self.df['Ticker'] == ticker]['Market Price'].iloc[0]
                        elif ticker in self.new_stocks['Ticker'].values:
                            prices[ticker] = self.new_stocks[self.new_stocks['Ticker'] == ticker]['Current Price'].iloc[0]
                except Exception as e:
                    st.warning(f"Error mengambil data untuk {ticker}: {str(e)}")
                    # Use existing price as fallback
                    if ticker in self.df['Ticker'].values:
                        prices[ticker] = self.df[self.df['Ticker'] == ticker]['Market Price'].iloc[0]
                    elif ticker in self.new_stocks['Ticker'].values:
                        prices[ticker] = self.new_stocks[self.new_stocks['Ticker'] == ticker]['Current Price'].iloc[0]
            
            # Update portfolio with real-time prices
            for idx, row in self.df.iterrows():
                ticker = row['Ticker']
                if ticker in prices:
                    self.df.at[idx, 'Market Price'] = prices[ticker]
            
            # Update new stocks with real-time prices
            for idx, row in self.new_stocks.iterrows():
                ticker = row['Ticker']
                if ticker in prices:
                    self.new_stocks.at[idx, 'Current Price'] = prices[ticker]
            
            # Recalculate market values
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
        return {
            'total_invested': df['Stock Value'].sum(),
            'total_market_value': df['Market Value'].sum(),
            'total_unrealized': df['Unrealized'].sum(),
            'return_pct': (df['Unrealized'].sum() / df['Stock Value'].sum()) * 100
        }
    
    def predict_price(self, stock, days=30):
        """Predict future prices using ML model"""
        if stock not in self.pm.simulated_data:
            return None, None, None
            
        data = self.pm.simulated_data[stock].copy()
        data['Days'] = (data['Date'] - data['Date'].min()).dt.days
        
        # Create features
        data['MA7'] = data['Price'].rolling(window=7).mean()
        data['MA30'] = data['Price'].rolling(window=30).mean()
        data = data.dropna()
        
        # Prepare data
        X = data[['Days', 'MA7', 'MA30']]
        y = data['Price']
        
        # Train model
        model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
        model.fit(X, y)
        
        # Generate future dates
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
        future_days = [(fd - data['Date'].min()).days for fd in future_dates]
        
        # Calculate moving averages for prediction
        last_ma7 = data['MA7'].iloc[-1]
        last_ma30 = data['MA30'].iloc[-1]
        future_ma7 = []
        future_ma30 = []
        
        # Sederhanakan perhitungan moving averages
        for i in range(days):
            # Untuk MA7
            if i < 7:
                future_ma7.append(np.mean(data['Price'].iloc[-(7-i):]))
            else:
                future_ma7.append(last_ma7)
                
            # Untuk MA30
            if i < 30:
                future_ma30.append(np.mean(data['Price'].iloc[-(30-i):]))
            else:
                future_ma30.append(last_ma30)
        
        # Predict
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
            
            # Get price trend
            trend = 0
            if stock in self.pm.simulated_data:
                data = self.pm.simulated_data[stock]
                if len(data) > 10:
                    trend = (data['Price'].iloc[-1] / data['Price'].iloc[-10] - 1) * 100
            
            # Recommendation logic
            if unrealized_pct < -15 or trend < -5:
                rec = 'Sell'
                reason = 'Significant loss & downward trend'
                urgency = 'High'
            elif unrealized_pct > 20 or trend > 8:
                rec = 'Buy More'
                reason = 'Strong performance & upward trend'
                urgency = 'Medium'
            elif unrealized_pct > 5 or trend > 3:
                rec = 'Hold/Buy'
                reason = 'Positive performance'
                urgency = 'Low'
            elif unrealized_pct < -5:
                rec = 'Hold/Sell'
                reason = 'Mild underperformance'
                urgency = 'Monitor'
            else:
                rec = 'Hold'
                reason = 'Stable performance'
                urgency = 'Low'
                
            recommendations.append({
                'Stock': stock,
                'Recommendation': rec,
                'Reason': reason,
                'Urgency': urgency,
                'Unrealized %': f"{unrealized_pct:.1f}%",
                '30d Trend %': f"{trend:.1f}%"
            })
            
        return pd.DataFrame(recommendations)


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
        
        # Add confidence interval
        if 'upper' in forecast and 'lower' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                y=forecast['upper'].tolist() + forecast['lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(100, 150, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Interval Keyakinan'
            ))
        
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
        # Get returns for each stock
        returns_data = self.calculate_stock_returns()
        
        # Create a DataFrame for returns
        returns_df = pd.DataFrame()
        for stock in self.pm.df['Stock']:
            if stock in returns_data:
                returns_df[stock] = returns_data[stock]['Return']
        
        # If there's no data, return None
        if returns_df.empty:
            return None
        
        # Get current market value for each stock to calculate weights
        total_value = self.pm.df['Market Value'].sum()
        weights = self.pm.df.set_index('Stock')['Market Value'] / total_value
        
        # Align weights with the columns in returns_df
        weights = weights[returns_df.columns].values
        
        # Calculate portfolio returns (weighted average)
        portfolio_returns = returns_df.dot(weights)
        return portfolio_returns.dropna()
    
    def portfolio_volatility(self, annualize=True):
        """Calculate portfolio volatility (standard deviation of returns)"""
        if self.portfolio_returns is None:
            return 0
        vol = self.portfolio_returns.std()
        if annualize:
            vol = vol * np.sqrt(252)  # annualized
        return vol
    
    def value_at_risk(self, confidence_level=0.95, method='historical'):
        """Calculate Value at Risk (VaR) for the portfolio"""
        if self.portfolio_returns is None:
            return 0
        if method == 'historical':
            # Historical VaR: the (1-confidence_level) percentile of returns
            return -np.percentile(self.portfolio_returns, 100*(1-confidence_level))
        else:
            # Parametric VaR (normal distribution)
            mean = self.portfolio_returns.mean()
            std = self.portfolio_returns.std()
            z_score = norm.ppf(1-confidence_level)
            return -(mean + z_score * std)
    
    def expected_shortfall(self, confidence_level=0.95):
        """Calculate Expected Shortfall (ES) for the portfolio"""
        if self.portfolio_returns is None:
            return 0
        var = self.value_at_risk(confidence_level, 'historical')
        # ES is the average of losses beyond VaR
        losses = -self.portfolio_returns
        return losses[losses > var].mean()
    
    def beta_analysis(self):
        """Calculate beta coefficients relative to market index (using JKSE)"""
        try:
            # Download market data (JKSE index)
            market = yf.download('^JKSE', period='1y')['Close'].pct_change().dropna()
            
            beta_values = {}
            for stock in self.pm.df['Stock']:
                if stock in self.pm.simulated_data:
                    stock_returns = self.pm.simulated_data[stock]['Price'].pct_change().dropna()
                    # Align dates
                    common_dates = stock_returns.index.intersection(market.index)
                    if len(common_dates) > 10:  # Minimum data points
                        cov_matrix = np.cov(
                            stock_returns.loc[common_dates], 
                            market.loc[common_dates]
                        )
                        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                        beta_values[stock] = beta
            return beta_values
        except:
            return {}
    
    def stress_test(self, scenario='crisis'):
        """Stress test portfolio under different market conditions"""
        scenarios = {
            'crisis': -0.30,  # 30% market decline
            'recession': -0.15,  # 15% market decline
            'bull': 0.20,  # 20% market rise
            'correction': -0.10  # 10% correction
        }
        
        if scenario not in scenarios:
            return None
        
        # Get beta values
        betas = self.beta_analysis()
        
        # Calculate expected returns under scenario
        scenario_return = scenarios[scenario]
        portfolio_impact = 0
        stock_impacts = {}
        
        for idx, row in self.pm.df.iterrows():
            stock = row['Stock']
            beta = betas.get(stock, 1.0)  # Default beta=1 if not available
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
        """Calculate diversification metrics for the portfolio"""
        # Get returns data
        returns_data = self.calculate_stock_returns()
        returns_df = pd.DataFrame()
        for stock, data in returns_data.items():
            returns_df[stock] = data['Return']
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Calculate average correlation
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        avg_correlation = corr_matrix.where(mask).mean().mean()
        
        # Calculate diversification ratio
        weighted_vol = self.pm.df['Market Value'] / self.pm.df['Market Value'].sum()
        weighted_vol = weighted_vol.values
        individual_vol = returns_df.std().values
        portfolio_vol = self.portfolio_volatility(annualize=False)
        
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
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = PortfolioManager()
    
    pm = st.session_state.portfolio
    analyzer = PortfolioAnalyzer(pm)
    visualizer = PortfolioVisualizer()
    
    st.title("📊 Dashboard Analisis Portofolio Lanjutan")
    st.caption("Alat interaktif untuk manajemen portofolio dan analisis saham")
    
    # Real-time price update section
    st.header("🔄 Data Pasar Real-time")
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
    
    # Portfolio Summary
    st.header("📈 Ringkasan Portofolio")
    summary = analyzer.portfolio_summary()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Investasi", f"Rp {summary['total_invested']:,.0f}")
    col2.metric("Nilai Saat Ini", f"Rp {summary['total_market_value']:,.0f}", 
                f"{summary['return_pct']:.2f}%")
    col3.metric("Keuntungan/Rugi Belum Direalisasi", f"Rp {summary['total_unrealized']:,.0f}", 
                f"{summary['return_pct']:.2f}%", delta_color="inverse")
    
    # Portfolio Charts
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(visualizer.portfolio_pie(pm.df), use_container_width=True)
    with col2:
        st.plotly_chart(visualizer.performance_bar(pm.df), use_container_width=True)
    
    # Stock Details with real-time data
    st.header("📋 Detail Saham Real-time")
    
    # Calculate performance metrics
    pm.df['Unrealized %'] = (pm.df['Unrealized'] / pm.df['Stock Value']) * 100
    pm.df['Daily Change'] = (pm.df['Market Price'] / pm.df['Avg Price'] - 1) * 100
    pm.df['Current Value'] = pm.df['Balance'] * pm.df['Market Price']
    
    # Format and display the dataframe
    formatted_df = pm.df[['Stock', 'Balance', 'Avg Price', 'Market Price', 
                          'Daily Change', 'Current Value', 'Unrealized', 'Unrealized %']].copy()
    
    # Format columns
    formatted_df['Avg Price'] = formatted_df['Avg Price'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Market Price'] = formatted_df['Market Price'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Current Value'] = formatted_df['Current Value'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Unrealized'] = formatted_df['Unrealized'].apply(lambda x: f"Rp {x:,.0f}")
    formatted_df['Daily Change'] = formatted_df['Daily Change'].apply(lambda x: f"{x:.2f}%")
    formatted_df['Unrealized %'] = formatted_df['Unrealized %'].apply(lambda x: f"{x:.2f}%")
    
    # Apply color coding
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
    
    styled_df = formatted_df.style.applymap(color_negative_red, 
                                           subset=['Daily Change', 'Unrealized', 'Unrealized %'])
    
    st.dataframe(styled_df, height=400, use_container_width=True)
    
    # Price Prediction
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
                    
                    # Simulate confidence interval
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
    
    # What If Simulation
    st.header("🎮 Analisis Skenario What If")
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
    
    # Buy/Sell Recommendations
    st.header("💡 Rekomendasi Beli/Jual")
    rec_df = analyzer.generate_recommendations()
    
    # Color coding for recommendations
    rec_colors = {
        'Sell': 'red',
        'Buy More': 'green',
        'Hold/Buy': 'lightgreen',
        'Hold/Sell': 'orange',
        'Hold': 'gray'
    }
    
    # Apply styling
    styled_rec = rec_df.style.apply(lambda x: [
        f"background-color: {rec_colors.get(v, 'white')}" for v in x
    ], subset=['Recommendation'])
    
    st.dataframe(styled_rec, height=400)
    
    # Portfolio Management
    st.header("⚙️ Manajemen Portofolio")
    
    # Add new stock
    st.subheader("Tambahkan Saham Baru")
    new_col1, new_col2 = st.columns([2, 1])
    
    with new_col1:
        selected_new = st.selectbox("Pilih Saham", pm.new_stocks['Stock'])
    
    with new_col2:
        shares = st.number_input("Jumlah Saham", min_value=1, value=100)
    
    if st.button("Tambahkan ke Portofolio", key='add_stock'):
        new_stock = pm.new_stocks[pm.new_stocks['Stock'] == selected_new].iloc[0]
        price = new_stock['Current Price']
        
        new_row = pd.DataFrame({
            'Stock': [selected_new],
            'Ticker': [new_stock['Ticker']],
            'Lot Balance': [shares / 100],
            'Balance': [shares],
            'Avg Price': [price],
            'Stock Value': [shares * price],
            'Market Price': [price],
            'Market Value': [shares * price],
            'Unrealized': [0]
        })
        
        pm.df = pd.concat([pm.df, new_row], ignore_index=True)
        st.success(f"Berhasil menambahkan {shares} saham {selected_new} ke portofolio!")
        st.session_state.portfolio = pm
    
    # Portfolio modifications
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
    
    # Remove stock
    if st.button("Hapus Saham Terpilih", type="primary"):
        if mod_stock:
            pm.df = pm.df[pm.df['Stock'] != mod_stock]
            st.success(f"Berhasil menghapus {mod_stock} dari portofolio!")
            st.session_state.portfolio = pm
    
    # Show current portfolio
    st.subheader("Portofolio Saat Ini")
    st.dataframe(pm.df[['Stock', 'Balance', 'Avg Price', 'Market Price', 'Unrealized']], 
                 height=300)

    # Risk Analysis Section
    st.header("📉 Analisis Risiko")
    risk_analyzer = RiskAnalyzer(pm)
    
    # Basic risk metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Volatilitas Portofolio (Tahunan)", f"{risk_analyzer.portfolio_volatility()*100:.2f}%")
    col2.metric("95% VaR (1-hari)", f"Rp {risk_analyzer.value_at_risk(0.95)*risk_analyzer.pm.df['Market Value'].sum():,.0f}")
    col3.metric("95% Expected Shortfall (1-hari)", f"Rp {risk_analyzer.expected_shortfall(0.95)*risk_analyzer.pm.df['Market Value'].sum():,.0f}")
    
    # Beta Analysis
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
    
    # Stress Testing
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
            
            # Show individual stock impacts
            impact_df = pd.DataFrame.from_dict(results['stock_impacts'], orient='index')
            impact_df = impact_df.reset_index().rename(columns={'index': 'Saham'})
            impact_df['Dampak %'] = (impact_df['Impact'] / pm.df.set_index('Stock')['Market Value']) * 100
            
            fig = px.bar(impact_df, x='Saham', y='Dampak %', 
                         title='Dampak per Saham',
                         color='Dampak %', 
                         color_continuous_scale='RdYlGn' if scenario == 'bull' else 'RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # Diversification Analysis
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
