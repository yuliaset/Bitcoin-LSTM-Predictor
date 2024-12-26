import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import timedelta

def fetch_data(ticker='BTC-USD', period='5y', interval='1d'):
    print(f"Fetching data for {ticker} with period '{period}' and interval '{interval}'...")
    data = yf.download(ticker, period=period, interval=interval)
    
    data = data[['Close']]  
    data.dropna(inplace=True)
    
    print(f"Successfully fetched data from {data.index.min().date()} to {data.index.max().date()} with {len(data)} records.")
    return data

# 2. Preprocess the Data
def preprocess_data(data, sequence_length=60, test_size=0.2, scaler_type='robust'):
    print("Starting data preprocessing...")
    
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_data = scaler.fit_transform(data)
    print(f"Data scaling with {scaler_type.capitalize()}Scaler completed.")

    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])  
    X, y = np.array(X), np.array(y)
    print(f"Total sequences created: {len(X)}")

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print("Data reshaped for LSTM.")

    split = int(X.shape[0] * (1 - test_size))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler

def build_model(input_shape):
    print("Building the LSTM model...")
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model compilation completed.")
    
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    print("Starting model training...")
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    print("Model training completed.")
    return history

def predict_and_plot(model, X_test, y_test, scaler, data, sequence_length=60, forecast_days=365):
    print("Making predictions on the test data...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_data = data[:-len(y_test)]
    test_data = data[-len(y_test):]

    print(f"Starting forecasting for the next {forecast_days} days...")
    last_sequence = X_test[-1]  
    future_predictions = []

    for i in range(forecast_days):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        future_predictions.append(next_pred[0, 0])

        last_sequence = np.append(last_sequence[:, 0], next_pred[0, 0])[-sequence_length:]
        last_sequence = last_sequence.reshape(sequence_length, 1)

        if (i + 1) % 50 == 0 or (i + 1) == forecast_days:
            print(f"Forecasted {i + 1}/{forecast_days} days")

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted_Close'])

    plt.figure(figsize=(14,7))
    plt.plot(train_data.index, train_data['Close'], label='Training Data')
    plt.plot(test_data.index, predictions, label='Test Predictions', color='green')
    plt.plot(future_df.index, future_df['Predicted_Close'], color='green')
    plt.title('Bitcoin Price Prediction and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
    print("Prediction and plotting completed.")

def main():
    TICKER = 'BTC-USD'
    PERIOD = '5y'
    INTERVAL = '1d'
    SEQUENCE_LENGTH = 60
    TEST_SIZE = 0.2
    EPOCHS = 50
    BATCH_SIZE = 32
    FORECAST_DAYS = 365
    SCALER_TYPE = 'robust'  

    try:
        data = fetch_data(ticker=TICKER, period=PERIOD, interval=INTERVAL)

        X_train, y_train, X_test, y_test, scaler = preprocess_data(
            data,
            sequence_length=SEQUENCE_LENGTH,
            test_size=TEST_SIZE,
            scaler_type=SCALER_TYPE
        )

        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.summary()

        history = train_model(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

        plt.figure(figsize=(10,4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        predict_and_plot(
            model, 
            X_test, 
            y_test, 
            scaler, 
            data, 
            sequence_length=SEQUENCE_LENGTH,
            forecast_days=FORECAST_DAYS
        )

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
