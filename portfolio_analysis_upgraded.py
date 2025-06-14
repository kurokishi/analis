import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import yfinance as yf
import os

# Disable oneDNN optimizations to avoid CUDA-related errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Parsing portfolio data
data = {
    'Stock': ['AADI', 'ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'LPPF', 'PGAS', 'PTBA', 'UNVR', 'WIIM'],
    'Lot Balance': [5.0, 17.0, 15.0, 30.0, 23.0, 11.0, 5.0, 10.0, 4.0, 60.0, 5.0],
    'Balance': [500, 1700, 1500, 3000, 2300, 1100, 500, 1000, 400, 6000, 500],
    'Avg Price': [7300, 2605, 1423, 1080, 1145, 4489, 1700, 1600, 2400, 1860, 871],
    'Stock Value': [3650000, 4428500, 2135000, 3240000, 2633500, 4938000, 850000, 1600000, 960000, 11162500, 435714],
    'Market Price': [7225, 2200, 3110, 905, 850, 4400, 1745, 1820, 2890, 1730, 835],
    'Market Value': [3612500, 3740000, 4665000, 2715000, 1955000, 4840000, 872500, 1820000, 1156000, 10380000, 417500],
    'Unrealized': [-37500, -688500, 2530000, -525000, -678500, -98000, 22500, 220000, 196000, -782500, -18215]
}
df = pd.DataFrame(data)

# Simulated fundamental data (replace with real data from yfinance or other APIs)
fundamental_data = pd.DataFrame({
    'Stock': df['Stock'],
    'P/E': [15.2, 8.5, 12.3, 10.1, 9.8, 7.5, 20.1, 11.2, 9.0, 25.3, 14.5],
    'P/B': [1.8, 1.2, 1.5, 1.0, 0.9, 1.3, 2.5, 1.4, 1.1, 3.2, 1.7],
    'ROE': [12.0, 15.5, 13.8, 10.2, 11.5, 18.0, 8.5, 14.2, 16.0, 20.5, 9.8],
    'D/E': [0.5, 0.8, 0.3, 1.2, 0.7, 0.4, 0.9, 0.6, 0.2, 0.1, 0.5],
    'Dividend Yield': [2.5, 4.0, 3.2, 2.8, 3.5, 5.0, 1.5, 3.0, 4.5, 2.0, 2.2],
    'ESG Score': [75, 60, 65, 55, 70, 50, 80, 68, 62, 85, 72]
})

# Simulated historical data with returns (replace with real data if available)
np.random.seed(42)
dates = pd.date_range(end='2025-05-31', periods=252, freq='B')
simulated_data = {}
for stock in df['Stock']:
    prices = np.cumprod(1 + np.random.normal(0, 0.01, 252)) * df[df['Stock'] == stock]['Market Price'].iloc[0]
    returns = np.diff(prices) / prices[:-1]
    simulated_data[stock] = pd.DataFrame({'Date': dates, 'Price': prices, 'Returns': np.append(np.nan, returns)})

# Simulated new stock recommendations with fundamentals
new_stocks = pd.DataFrame({
    'Stock': ['TLKM', 'BBCA', 'BMRI', 'ASII'],
    'Sector': ['Telecom', 'Banking', 'Banking', 'Automotive'],
    'Dividend Yield': [4.5, 3.2, 3.8, 2.9],
    'Growth Rate': [8.0, 10.0, 9.5, 7.0],
    'Current Price': [3500, 9500, 6000, 4500],
    'P/E': [14.5, 18.2, 16.0, 12.8],
    'P/B': [1.6, 2.3, 1.9, 1.4],
    'ESG Score': [78, 82, 80, 75]
})

# Streamlit app
st.title("Advanced Portfolio Analysis Tool")

# Portfolio Analysis
st.header("Portfolio Analysis")
total_invested = df['Stock Value'].sum()
total_market_value = df['Market Value'].sum()
total_unrealized = df['Unrealized'].sum()
st.write(f"**Total Invested Value**: Rp {total_invested:,.0f}")
st.write(f"**Total Market Value**: Rp {total_market_value:,.0f}")
st.write(f"**Total Unrealized Gain/Loss**: Rp {total_unrealized:,.0f} ({(total_unrealized/total_invested)*100:.2f}%)")

# Fundamental Analysis
st.subheader("Fundamental Analysis")
st.dataframe(fundamental_data.merge(df[['Stock', 'Market Price', 'Unrealized']], on='Stock'))

# Risk Metrics
st.subheader("Risk Metrics")
returns_matrix = pd.DataFrame({stock: simulated_data[stock]['Returns'] for stock in df['Stock']})
cov_matrix = returns_matrix.cov() * 252
weights = df['Market Value'] / total_market_value
portfolio_return = np.sum(returns_matrix.mean() * weights) * 252
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
sharpe_ratio = (portfolio_return - 0.05) / portfolio_volatility  # Assuming 5% risk-free rate
st.write(f"**Portfolio Expected Return**: {portfolio_return:.2%}")
st.write(f"**Portfolio Volatility**: {portfolio_volatility:.2%}")
st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")

# Portfolio Composition
fig = px.pie(df, values='Market Value', names='Stock', title='Portfolio Composition')
st.plotly_chart(fig)

# Advanced Price Prediction (LSTM)
st.header("Stock Price Prediction (AI)")
selected_stock = st.selectbox("Select Stock for Prediction", df['Stock'])
if selected_stock:
    stock_data = simulated_data[selected_stock][['Date', 'Price']]
    prices = stock_data['Price'].values
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    
    # Prepare data for LSTM
    time_steps = 10
    X, y = [], []
    for i in range(len(scaled_prices) - time_steps):
        X.append(scaled_prices[i:i+time_steps])
        y.append(scaled_prices[i+time_steps])
    X, y = np.array(X), np.array(y)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    # Predict future prices
    future_days = 30
    last_sequence = scaled_prices[-time_steps:]
    predictions = []
    for _ in range(future_days):
        pred = model.predict(last_sequence.reshape(1, time_steps, 1), verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = [stock_data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})
    hist_df = stock_data[['Date', 'Price']].rename(columns={'Price': 'Predicted Price'})
    plot_df = pd.concat([hist_df, pred_df])
    
    fig = px.line(plot_df, x='Date', y='Predicted Price', title=f"Price Prediction for {selected_stock} (LSTM)")
    st.plotly_chart(fig)
    st.write(f"Predicted Price in 30 Days: Rp {predictions[-1]:,.0f}")

# Monte Carlo Simulation
st.header("Monte Carlo Simulation")
num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
sim_horizon = st.slider("Simulation Horizon (Years)", 1, 10, 5)
sim_results = []
for _ in range(num_simulations):
    sim_returns = np.random.multivariate_normal(returns_matrix.mean() * 252, cov_matrix, sim_horizon * 252)
    sim_prices = total_market_value * np.cumprod(1 + sim_returns.dot(weights))
    sim_results.append(sim_prices[-1])
sim_results = np.array(sim_results)
st.write(f"**Expected Portfolio Value in {sim_horizon} Years**: Rp {np.mean(sim_results):,.0f}")
st.write(f"**5% Value-at-Risk (VaR)**: Rp {(total_market_value - np.percentile(sim_results, 5)):,.0f}")

fig = px.histogram(sim_results, title=f"Portfolio Value Distribution in {sim_horizon} Years")
st.plotly_chart(fig)

# Portfolio Optimization
st.header("Portfolio Optimization")
st.subheader("Mean-Variance Optimization")
def portfolio_performance(weights, returns, cov_matrix, risk_free_rate=0.05):
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return port_return, port_vol, (port_return - risk_free_rate) / port_vol

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.05):
    return -portfolio_performance(weights, returns, cov_matrix, risk_free_rate)[2]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(df)))
init_guess = np.array([1./len(df)] * len(df))
opt_result = minimize(negative_sharpe_ratio, init_guess, args=(returns_matrix, cov_matrix),
                     method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = opt_result.x
opt_port = pd.DataFrame({'Stock': df['Stock'], 'Optimal Weight': opt_weights})
st.dataframe(opt_port)
fig = px.pie(opt_port, values='Optimal Weight', names='Stock', title='Optimized Portfolio')
st.plotly_chart(fig)

# Stress Testing
st.header("Stress Testing")
scenario = st.selectbox("Select Market Scenario", ['Base Case', '2008 Crisis', 'High Inflation', 'Tech Crash'])
scenario_params = {
    'Base Case': {stock: 0 for stock in df['Stock']},
    '2008 Crisis': {stock: -0.3 for stock in df['Stock']},  # 30% drop
    'High Inflation': {stock: -0.1 for stock in df['Stock']},  # 10% drop
    'Tech Crash': {stock: -0.5 if stock in ['AADI', 'WIIM'] else -0.2 for stock in df['Stock']}  # Tech-heavy stocks drop more
}
sim_df = df.copy()
for stock in df['Stock']:
    idx = sim_df[sim_df['Stock'] == stock].index
    sim_df.loc[idx, 'Market Price'] *= (1 + scenario_params[scenario][stock])
    sim_df.loc[idx, 'Market Value'] = sim_df.loc[idx, 'Balance'] * sim_df.loc[idx, 'Market Price']
    sim_df.loc[idx, 'Unrealized'] = sim_df.loc[idx, 'Market Value'] - sim_df.loc[idx, 'Stock Value']
st.write(f"**Scenario: {scenario}**")
st.write(f"**New Total Market Value**: Rp {sim_df['Market Value'].sum():,.0f}")
st.dataframe(sim_df[['Stock', 'Market Price', 'Market Value', 'Unrealized']])

# Buy/Sell Recommendations
st.header("Buy/Sell Recommendations")
recommendations = []
for _, row in df.iterrows():
    fundamentals = fundamental_data[fundamental_data['Stock'] == row['Stock']].iloc[0]
    unrealized_pct = (row['Unrealized'] / row['Stock Value']) * 100
    if fundamentals['P/E'] > 20 or fundamentals['P/B'] > 2.5 or unrealized_pct < -15:
        recommendations.append({'Stock': row['Stock'], 'Recommendation': 'Sell', 'Reason': 'High valuation or significant loss'})
    elif fundamentals['P/E'] < 10 and fundamentals['P/B'] < 1.5 and fundamentals['ROE'] > 12:
        recommendations.append({'Stock': row['Stock'], 'Recommendation': 'Buy', 'Reason': 'Undervalued with strong fundamentals'})
    else:
        recommendations.append({'Stock': row['Stock'], 'Recommendation': 'Hold', 'Reason': 'Stable performance'})
rec_df = pd.DataFrame(recommendations)
st.dataframe(rec_df)

# New Stock Recommendations
st.header("New Stock Recommendations")
st.write("Based on sector, dividend yield, growth potential, and ESG scores:")
st.dataframe(new_stocks)
selected_new_stock = st.selectbox("Select New Stock to Add", new_stocks['Stock'])
if st.button("Add to Portfolio"):
    new_row = pd.DataFrame({
        'Stock': [selected_new_stock],
        'Lot Balance': [1.0],
        'Balance': [100],
        'Avg Price': [new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Stock Value': [100 * new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Market Price': [new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Market Value': [100 * new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Unrealized': [0]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    st.write("Stock added to portfolio!")
    st.dataframe(df)

# Save updated portfolio (simulated)
st.write("Updated portfolio (simulated save):")
st.dataframe(df)