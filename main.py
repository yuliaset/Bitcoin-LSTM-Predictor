import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker = 'BTC-USD'
start_date = '2015-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
btc_data = yf.download(ticker, start=start_date, end=end_date)
btc_data.dropna(inplace=True)

features = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].values)
        y.append(data['Close'].values[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_features, SEQ_LENGTH)

split_index = X.shape[0] - 365  
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = models.Sequential([
    layers.LSTM(100, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    layers.LSTM(100, return_sequences=False),
    layers.Dense(50, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

EPOCHS = 20
BATCH_SIZE = 32
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=False
)

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

last_sequence = scaled_features[-SEQ_LENGTH:].values
last_sequence = last_sequence.reshape((1, SEQ_LENGTH, scaled_features.shape[1]))

predictions = []
current_sequence = last_sequence.copy()

for _ in range(365):
    pred = model.predict(current_sequence)[0][0]
    predictions.append(pred)
    
    new_entry = current_sequence[0, 1:, :].tolist()  
    last_features = current_sequence[0, -1, :].tolist()
    last_features[3] = pred  
    new_entry.append(last_features)
    current_sequence = np.array([new_entry])

close_scaler = scaler.scale_[3]
close_min = scaler.min_[3]
predictions_actual = (np.array(predictions) - close_min) / close_scaler

last_date = scaled_features.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]

historical_close = btc_data['Close']
future_close = pd.Series(predictions_actual, index=future_dates)
combined_close = pd.concat([historical_close, future_close])

plt.figure(figsize=(14,7))
plt.plot(historical_close, label='Historical Close Prices')
plt.plot(future_close, label='Predicted Close Prices', linestyle='--')
plt.title('Bitcoin Price Prediction for the Next Year')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()