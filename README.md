# AAPL Stock Price Prediction using LSTM

Deep learning model to predict Apple stock prices using LSTM neural network.

## Project Overview
Built an end-to-end deep learning pipeline to forecast AAPL stock prices using historical data from Yahoo Finance.

## Tech Stack
- Python, Pandas, NumPy
- TensorFlow, Keras
- LSTM Neural Network
- Scikit-learn (MinMaxScaler)
- Matplotlib
- yFinance API

## Features
- Real-time stock data download using yFinance
- Data normalization using MinMaxScaler
- 2-layer LSTM model with Dropout regularization
- Actual vs Predicted price visualization
- RMSE: 5.85

## Model Architecture
- LSTM Layer 1: 50 units
- Dropout: 0.2
- LSTM Layer 2: 50 units
- Dropout: 0.2
- Dense Layer: 25 units
- Output Layer: 1 unit

## How to Run
pip install tensorflow yfinance scikit-learn matplotlib pandas numpy
python stock_model.py

## Results
- Trained on 1006 days of AAPL stock data (2020-2024)
- Sequence length: 60 days
- Achieved RMSE of 5.85
