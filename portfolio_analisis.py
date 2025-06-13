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

# ======================
# DATA MANAGEMENT MODULE
# ======================
class PortfolioManager:
    def __init__(self):
        self.df = self.load_portfolio()
        self.simulated_data = self.generate_historical_data()
        self.new_stocks = self.get_new_stocks()
        
    @staticmethod
    def load_portfolio():
        """Initialize portfolio data"""
        return pd.DataFrame({
            'Stock': ['AADI', 'ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'LPPF', 'PGAS', 'PTBA', 'UNVR', 'WIIM'],
            'Lot Balance': [5.0, 17.0, 15.0, 30.0, 23.0, 11.0, 5.0, 10.0, 4.0, 60.0, 5.0],
            'Balance': [500, 1700, 1500, 3000, 2300, 1100, 500, 1000, 400, 6000, 500],
            'Avg Price': [7300, 2605, 1423, 1080, 1145, 4489, 1700, 1600, 2400, 1860, 871],
            'Stock Value': [3650000, 4428500, 2135000, 3240000, 2633500, 4938000, 850000, 1600000, 960000, 11162500, 435714],
            'Market Price': [7225, 2200, 3110, 905, 850, 4400, 1745, 1820, 2890, 1730, 835],
            'Market Value': [3612500, 3740000, 4665000, 2715000, 1955000, 4840000, 872500, 1820000, 1156000, 10380000, 417500],
            'Unrealized': [-37500, -688500, 2530000, -525000, -678500, -98000, 22500, 220000, 196000, -782500, -18215]
        })
    
    def generate_historical_data(self):
        """Generate realistic historical price data"""
        np.random.seed(42)
        dates = pd.date_range(end='2025-05-31', periods=100, freq='D')
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
            'Sector': ['Telecom', 'Banking', 'Banking', 'Automotive'],
            'Dividend Yield': [4.5, 3.2, 3.8, 2.9],
            'Growth Rate': [8.0, 10.0, 9.5, 7.0],
            'Current Price': [3500, 9500, 6000, 4500],
            'Risk Level': ['Low', 'Medium', 'Medium', 'High']
        })


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
        
        # PERBAIKAN: Menggunakan pendekatan yang lebih sederhana untuk moving averages
        for i in range(days):
            # Untuk MA7
            if i < 7:
                # Gunakan data historis yang tersedia
                future_ma7.append(np.mean(data['Price'].iloc[-(7-i):]))
            else:
                # Gunakan nilai terakhir
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
                     title='Portfolio Composition', hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    @staticmethod
    def performance_bar(df):
        """Create unrealized gain/loss bar chart"""
        df = df.copy()
        df['Color'] = df['Unrealized'].apply(lambda x: 'green' if x >= 0 else 'red')
        fig = px.bar(df, x='Stock', y='Unrealized', color='Color',
                     title='Unrealized Gain/Loss by Stock',
                     labels={'Unrealized': 'Gain/Loss (Rp)'})
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
                      title=f"Price Forecast for {stock}",
                      labels={'Value': 'Price (Rp)'})
        
        # Add confidence interval
        if 'confidence' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                y=forecast['upper'].tolist() + forecast['lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(100, 150, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        
        return fig


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
    
    st.title("📊 Advanced Portfolio Analysis Dashboard")
    st.caption("Interactive tool for portfolio management and stock analysis")
    
    # Portfolio Summary
    st.header("📈 Portfolio Summary")
    summary = analyzer.portfolio_summary()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Invested", f"Rp {summary['total_invested']:,.0f}")
    col2.metric("Current Value", f"Rp {summary['total_market_value']:,.0f}", 
                f"{summary['return_pct']:.2f}%")
    col3.metric("Unrealized P&L", f"Rp {summary['total_unrealized']:,.0f}", 
                f"{summary['return_pct']:.2f}%", delta_color="inverse")
    
    # Portfolio Charts
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(visualizer.portfolio_pie(pm.df), use_container_width=True)
    with col2:
        st.plotly_chart(visualizer.performance_bar(pm.df), use_container_width=True)
    
    # Stock Details
    st.subheader("📋 Stock Details")
    # PERBAIKAN: Menghitung kolom Unrealized %
    pm.df['Unrealized %'] = (pm.df['Unrealized'] / pm.df['Stock Value']) * 100
    st.dataframe(pm.df[['Stock', 'Balance', 'Avg Price', 'Market Price', 
                        'Unrealized', 'Unrealized %']].sort_values('Unrealized', ascending=False),
                 height=300)
    
    # Price Prediction
    st.header("🔮 AI Price Prediction")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_stock = st.selectbox("Select Stock", pm.df['Stock'])
        days = st.slider("Forecast Period (days)", 7, 90, 30)
        
    with col2:
        if selected_stock:
            with st.spinner("Generating forecast..."):
                dates, predictions, last_pred = analyzer.predict_price(selected_stock, days)
                
                if predictions is not None:
                    current_price = pm.df[pm.df['Stock'] == selected_stock]['Market Price'].iloc[0]
                    change_pct = (last_pred / current_price - 1) * 100
                    
                    st.metric(f"Predicted Price in {days} days", 
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
    st.header("🎮 What If Scenario Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        sim_stock = st.selectbox("Select Stock", pm.df['Stock'], key='sim_stock')
        price_change = st.slider("Price Change (%)", -50.0, 50.0, 10.0, key='price_slider')
    
    with col2:
        if sim_stock:
            result = analyzer.what_if_simulation(sim_stock, price_change)
            if result:
                current_value = pm.df['Market Value'].sum()
                new_value = result['new_total_market']
                change = (new_value - current_value) / current_value * 100
                
                st.metric("Portfolio Value Impact", 
                         f"Rp {new_value:,.0f}",
                         f"{change:.2f}%")
                
                st.metric("Unrealized P&L Impact", 
                         f"Rp {result['new_unrealized']:,.0f}",
                         f"{(result['new_unrealized'] - summary['total_unrealized'])/summary['total_invested']*100:.2f}%")
    
    # Buy/Sell Recommendations
    st.header("💡 Trading Recommendations")
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
    st.header("⚙️ Portfolio Management")
    
    # Add new stock
    st.subheader("Add New Stock")
    new_col1, new_col2 = st.columns([2, 1])
    
    with new_col1:
        selected_new = st.selectbox("Select Stock", pm.new_stocks['Stock'])
    
    with new_col2:
        shares = st.number_input("Shares", min_value=1, value=100)
    
    if st.button("Add to Portfolio", key='add_stock'):
        new_stock = pm.new_stocks[pm.new_stocks['Stock'] == selected_new].iloc[0]
        price = new_stock['Current Price']
        
        new_row = pd.DataFrame({
            'Stock': [selected_new],
            'Lot Balance': [shares / 100],
            'Balance': [shares],
            'Avg Price': [price],
            'Stock Value': [shares * price],
            'Market Price': [price],
            'Market Value': [shares * price],
            'Unrealized': [0]
        })
        
        pm.df = pd.concat([pm.df, new_row], ignore_index=True)
        st.success(f"Added {shares} shares of {selected_new} to portfolio!")
        st.session_state.portfolio = pm
    
    # Portfolio modifications
    st.subheader("Modify Holdings")
    mod_stock = st.selectbox("Select Stock", pm.df['Stock'], key='mod_stock')
    
    if mod_stock:
        current_balance = pm.df[pm.df['Stock'] == mod_stock]['Balance'].iloc[0]
        new_balance = st.number_input("New Share Amount", min_value=0, value=int(current_balance))
        
        if st.button("Update Shares"):
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
                
                st.success(f"Updated {mod_stock} to {new_balance} shares!")
                st.session_state.portfolio = pm
    
    # Remove stock
    if st.button("Remove Selected Stock", type="primary"):
        if mod_stock:
            pm.df = pm.df[pm.df['Stock'] != mod_stock]
            st.success(f"Removed {mod_stock} from portfolio!")
            st.session_state.portfolio = pm
    
    # Show current portfolio
    st.subheader("Current Portfolio")
    st.dataframe(pm.df[['Stock', 'Balance', 'Avg Price', 'Market Price', 'Unrealized']], 
                 height=300)

if __name__ == "__main__":
    main()
