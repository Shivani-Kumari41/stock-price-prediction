import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Download Stock Data
print("Downloading stock data...")
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
df = df[['Close']]
print(f"Data downloaded! Shape: {df.shape}")
print(df.head())

# MinMaxScaler - normalize data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)
print("Data normalized!")

# Create sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
print(f"Sequences created! X shape: {X.shape}, y shape: {y.shape}")

# Train test split - 80% train, 20% test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM - needs 3D input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train model
print("Training model...")
history = model.fit(X_train, y_train, 
                    batch_size=32, 
                    epochs=10, 
                    validation_split=0.1,
                    verbose=1)

print("Model trained!")


# Predictions
y_pred = model.predict(X_test)

# Inverse transform - wapas original scale pe laao
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# RMSE calculate karo
rmse = np.sqrt(np.mean((y_pred_actual - y_test_actual) ** 2))
print(f"RMSE: {rmse:.2f}")

# Chart banao
plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label='Actual Price', color='blue')
plt.plot(y_pred_actual, label='Predicted Price', color='orange')
plt.title('AAPL Stock Price - Actual vs Predicted')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig('stock_prediction.png')
plt.show()
print("Chart saved!")
