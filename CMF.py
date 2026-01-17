import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📈 Cryptocurrency Price Forecasting App")

tab1, tab2 = st.tabs(["📄 Dataset Analysis", "📊 Forecasting"])

# -------------------------------------------------
with tab1:
    st.header("Dataset Overview")
    st.markdown("""
    **crypto-markets.csv** contains historical cryptocurrency market data.

    **Target Variable:** `close` price  
    **Frequency:** Daily  
    **Use Cases:**  
    - Price forecasting  
    - Volatility modeling  
    - Risk analysis  
    """)

# -------------------------------------------------
with tab2:
    uploaded_file = st.sidebar.file_uploader("Upload crypto-markets.csv", type=["csv"])
    if not uploaded_file:
        st.warning("Upload dataset to continue")
        st.stop()

    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])

    st.sidebar.subheader("Filters")
    symbol = st.sidebar.selectbox("Cryptocurrency", sorted(df['symbol'].unique()))

    df = df[df['symbol'] == symbol].copy()
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    st.subheader(f"{symbol} Closing Price")
    st.write(df[['close']].head())

    fig = px.line(df, y='close', title="Closing Price Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Stationarity ----------------
    st.subheader("ADF Stationarity Test")

    def adf_test(series):
        res = adfuller(series.dropna())
        return {
            "ADF Statistic": res[0],
            "p-value": res[1],
            "Critical Values": res[4]
        }

    st.write(adf_test(df['close']))

    diff_series = df['close'].diff().dropna()
    st.write("After First Differencing")
    st.write(adf_test(diff_series))

    # ---------------- Train/Test ------------------
    train = df['close'][:-90]
    test = df['close'][-90:]

    # ---------------- ARIMA -----------------------
    if st.button("Run ARIMA"):
        model = ARIMA(train, order=(5,1,0)).fit()
        forecast = model.forecast(90)

        rmse = np.sqrt(mean_squared_error(test, forecast))
        r2 = r2_score(test, forecast)

        st.success(f"RMSE: {rmse:.2f} | R²: {r2:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
        fig.add_trace(go.Scatter(x=test.index, y=forecast, name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- GARCH -----------------------
    if st.button("Run GARCH Volatility"):
        returns = 100 * df['close'].pct_change().dropna()
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=90)
        vol = forecast.variance.values[-1]

        fig = px.line(vol, title="90-Day Forecasted Volatility")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- LSTM ------------------------
    if st.button("Run LSTM"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['close']])

        def create_seq(data, step=30):
            X, y = [], []
            for i in range(len(data)-step):
                X.append(data[i:i+step])
                y.append(data[i+step])
            return np.array(X), np.array(y)

        X, y = create_seq(scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        X_train, X_test = X[:-90], X[-90:]
        y_train, y_test = y[:-90], y[-90:]

        model = Sequential([
            LSTM(50, input_shape=(30,1)),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        preds = model.predict(X_test)
        preds = scaler.inverse_transform(preds)
        y_test = scaler.inverse_transform(y_test.reshape(-1,1))

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        st.success(f"LSTM RMSE: {rmse:.2f} | R²: {r2:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test.flatten(), name="Actual"))
        fig.add_trace(go.Scatter(y=preds.flatten(), name="Predicted"))
        st.plotly_chart(fig, use_container_width=True)
