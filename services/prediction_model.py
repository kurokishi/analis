# services/prediction_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def prepare_lstm_data(df, time_steps=60):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def predict_next_price(df, time_steps=60):
    X, y, scaler = prepare_lstm_data(df, time_steps)
    model = train_lstm_model(X, y)

    last_sequence = df[['Close']].values[-time_steps:]
    scaled_sequence = scaler.transform(last_sequence)
    X_pred = np.array([scaled_sequence])
    X_pred = X_pred.reshape((1, time_steps, 1))

    predicted_scaled = model.predict(X_pred, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return float(predicted_price[0, 0])

def predict_future_prices(df, days_ahead=7, time_steps=60):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y, _ = prepare_lstm_data(df, time_steps)
    model = train_lstm_model(X, y)

    predictions = []
    last_input = data_scaled[-time_steps:].reshape(1, time_steps, 1)

    for _ in range(days_ahead):
        next_pred_scaled = model.predict(last_input, verbose=0)
        predictions.append(next_pred_scaled[0, 0])
        last_input = np.append(last_input[:, 1:, :], [[next_pred_scaled]], axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predictions).flatten()
    return predicted_prices
