# =========================================================
# Cryptocurrency Market Forecasting App
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================================================
# Streamlit Configuration
# =========================================================
st.set_page_config(page_title="Crypto Forecasting", layout="wide")
st.title("📈 Cryptocurrency Market Forecasting App")

tab1, tab2 = st.tabs(["📄 Dataset Overview", "📊 Forecasting"])

# =========================================================
# Safe ADF Test Function (NO CRASH)
# =========================================================
def safe_adf_test(series):
    series = series.dropna()

    if len(series) < 10:
        return {"Status": "❌ Not enough data points for ADF test"}

    if series.nunique() == 1:
        return {"Status": "❌ Series is constant (ADF not applicable)"}

    result = adfuller(series, autolag="AIC")

    return {
        "ADF Statistic": round(result[0], 4),
        "p-value": round(result[1], 6),
        "Lags Used": result[2],
        "Observations": result[3],
        "Critical Values": result[4],
        "Stationary": "✅ Yes" if result[1] < 0.05 else "❌ No"
    }

# =========================================================
# TAB 1 : Dataset Overview
# =========================================================
with tab1:
    st.header("Dataset Description")
    st.markdown("""
    **crypto-markets.csv** contains daily historical cryptocurrency data.

    **Important Columns**
    - `date`
    - `symbol`
    - `open`, `high`, `low`, `close`
    - `volume`, `market_cap`

    **Target Variable:** `close` price  
    **Models Used:** ARIMA, GARCH, LSTM  
    **Objective:** Price & volatility forecasting
    """)

# =========================================================
# TAB 2 : Forecasting
# =========================================================
with tab2:
    uploaded_file = st.sidebar.file_uploader(
        "Upload crypto-markets.csv", type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Please upload the dataset to continue.")
        st.stop()

    # ---------------- Load Data ----------------
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])

    st.sidebar.subheader("Filters")
    symbol = st.sidebar.selectbox(
        "Cryptocurrency Symbol",
        sorted(df['symbol'].unique())
    )

    df = df[df['symbol'] == symbol].copy()
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # ---------------- Visualization -------------
    st.subheader(f"{symbol} – Closing Price")
    st.write(df[['close']].head())

    fig_price = px.line(
        df, y='close',
        title="Closing Price Over Time"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ---------------- Stationarity ---------------
    st.subheader("ADF Stationarity Test")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Series")
        st.write(safe_adf_test(df['close']))

    with col2:
        st.markdown("### First Differenced Series")
        diff_series = df['close'].diff()
        st.write(safe_adf_test(diff_series))

    st.info(
        "ADF Test: p-value < 0.05 ⇒ Stationary series"
    )

    # ---------------- Train/Test Split ----------
    train = df['close'][:-90]
    test = df['close'][-90:]

    # =====================================================
    # ARIMA MODEL
    # =====================================================
    st.subheader("ARIMA Price Forecast")

    if st.button("Run ARIMA"):
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=90)

        rmse = np.sqrt(mean_squared_error(test, forecast))
        r2 = r2_score(test, forecast)

        st.success(f"RMSE: {rmse:.2f} | R²: {r2:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
        fig.add_trace(go.Scatter(x=test.index, y=forecast, name="Forecast"))
        fig.update_layout(title="ARIMA Forecast")
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # GARCH MODEL
    # =====================================================
    st.subheader("GARCH Volatility Forecast")

    if st.button("Run GARCH"):
        returns = 100 * df['close'].pct_change().dropna()

        if returns.nunique() > 1:
            garch_model = arch_model(
                returns, vol='Garch', p=1, q=1
            )
            res = garch_model.fit(disp="off")
            forecast = res.forecast(horizon=90)
            volatility = forecast.variance.values[-1]

            fig = px.line(
                volatility,
                title="90-Day Forecasted Volatility"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Returns are constant. GARCH cannot be applied.")

    # =====================================================
    # LSTM MODEL
    # =====================================================
    st.subheader("LSTM Price Forecast")

    if st.button("Run LSTM"):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['close']])

        def create_sequences(data, look_back=30):
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:i + look_back])
                y.append(data[i + look_back])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        X_train, X_test = X[:-90], X[-90:]
        y_train, y_test = y[:-90], y[-90:]

        model = Sequential()
        model.add(LSTM(50, input_shape=(30, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=16,
            verbose=0
        )

        predictions = model.predict(X_test)
        predictions = scaler.i
