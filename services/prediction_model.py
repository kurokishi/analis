#services/prediction_model.py

import numpy as np
import pandas as pd 
from sklearn.linear_model 
import LinearRegression from sklearn.metrics 
import mean_squared_error from sklearn.preprocessing 
import StandardScaler

def prepare_regression_data(df, window=5):
    df = df[['Close']].copy() 
    df['target'] = df['Close'].shift(-1) 
    df.dropna(inplace=True) 
    X = [] 
    y = [] 
    for i in range(window, len(df)):
        X.append(df['Close'].iloc[i - window:i].values) 
        y.append(df['target'].iloc[i]) 
    return np.array(X), np.array(y)

def predict_next_price(df, window=5): 
    X, y = prepare_regression_data(df, window) 
    model = LinearRegression() 
    model.fit(X, y) 
    last_window = df['Close'].iloc[-window:].values.reshape(1, -1) 
    return float(model.predict(last_window)[0])

def predict_future_prices(df, days_ahead=7, window=5): 
    data = df['Close'].tolist() 
    predictions = [] 
    model = LinearRegression()

for _ in range(days_ahead):
    if len(data) < window:
        break
    X = [data[-window:]]
    model.fit(*prepare_regression_data(pd.DataFrame(data, columns=['Close']), window))
    pred = model.predict(X)[0]
    predictions.append(pred)
    data.append(pred)

return np.array(predictions)

