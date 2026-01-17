import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ================== PAGE CONFIG ==================
st.set_page_config(layout="wide")
st.title("📈 Cryptocurrency Market Forecasting App")

tab1, tab2 = st.tabs(["📄 Project Summary", "📊 Forecasting App"])

# ================== HELPERS ==================
def detect_crypto_column(df):
    for col in ["symbol", "coin", "name", "id", "asset"]:
        if col in df.columns:
            return col
    return None

def detect_date_column(df):
    for col in ["date", "timestamp", "time", "datetime"]:
        if col in df.columns:
            return col
    return None

# ================== TAB 1 ==================
with tab1:
    st.markdown("""
    ### 🔍 Objective
    Forecast cryptocurrency market behavior using time-series models.

    ### 🧠 Models
    - ARIMA
    - SARIMA
    - Exponential Smoothing
    - ARCH / GARCH
    - LSTM

    ### 📈 Output
    - 90-step forecast
    - Seasonality & trend analysis
    - Volatility modeling
    """)

# ================== TAB 2 ==================
with tab2:

    uploaded_file = st.sidebar.file_uploader(
        "Upload crypto-markets.csv",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

    # ---------- LOAD DATA ----------
    df = pd.read_csv(uploaded_file)

    # ---------- DETECT DATE COLUMN ----------
    date_col = detect_date_column(df)
    if date_col is None:
        st.error("No date column found. Expected: date / timestamp / time")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df.dropna(subset=[date_col], inplace=True)
    df.sort_values(date_col, inplace=True)

    # ---------- DETECT CRYPTO COLUMN ----------
    crypto_col = detect_crypto_column(df)
    if crypto_col is None:
        st.error(
            "No crypto identifier column found.\n"
            "Expected one of: symbol, coin, name, id, asset"
        )
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    st.sidebar.success(f"Detected crypto column: `{crypto_col}`")
    st.sidebar.success(f"Detected date column: `{date_col}`")

    # ---------- FILTERS ----------
    st.sidebar.subheader("Filters")

    crypto_value = st.sidebar.selectbox(
        "Cryptocurrency",
        sorted(df[crypto_col].astype(str).unique())
    )

    filtered_df = df[df[crypto_col].astype(str) == crypto_value].copy()
    filtered_df.set_index(date_col, inplace=True)

    # ---------- NUMERIC COLUMNS ----------
    numeric_cols = filtered_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns available for forecasting.")
        st.stop()

    target_col = st.sidebar.selectbox(
        "Target Variable",
        numeric_cols
    )

    series = filtered_df[target_col].dropna()

    if len(series) < 120:
        st.warning("Not enough data points for reliable forecasting.")
        st.stop()

    # ---------- PREVIEW ----------
    st.subheader(f"{crypto_value} — {target_col.upper()}")
    st.write(series.head())

    # ---------- TIME SERIES ----------
    fig = px.line(
        series,
        title=f"{crypto_value} {target_col.upper()} Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- DECOMPOSITION ----------
    st.subheader("Seasonal Decomposition")
    decomposition = sm.tsa.seasonal_decompose(
        series,
        model="additive",
        period=30
    )

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"]
    )

    fig.add_trace(go.Scatter(y=decomposition.observed), 1, 1)
    fig.add_trace(go.Scatter(y=decomposition.trend), 2, 1)
    fig.add_trace(go.Scatter(y=decomposition.seasonal), 3, 1)
    fig.add_trace(go.Scatter(y=decomposition.resid), 4, 1)

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- ADF TEST ----------
    st.subheader("ADF Stationarity Test")
    adf_result = adfuller(series)
    st.write({
        "ADF Statistic": adf_result[0],
        "p-value": adf_result[1],
        "Lags Used": adf_result[2]
    })

    # ---------- TRAIN / TEST ----------
    train = series[:-90]
    test = series[-90:]

    # ================== ARIMA ==================
    if st.button("Run ARIMA Forecast"):
        model = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = model.forecast(90)

        st.write(f"RMSE: {np.sqrt(mean_squared_error(test, forecast)):.2f}")
        st.write(f"R²: {r2_score(test, forecast):.4f}")

    # ================== SARIMA ==================
    if st.button("Run SARIMA Forecast"):
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7)
        ).fit()

        forecast = model.get_forecast(90).predicted_mean
        st.write(f"RMSE: {np.sqrt(mean_squared_error(test, forecast)):.2f}")

    # ================== EXP SMOOTH ==================
    if st.button("Run Exponential Smoothing"):
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=30
        ).fit()

        forecast = model.forecast(90)
        st.write(f"RMSE: {np.sqrt(mean_squared_error(test, forecast)):.2f}")

    # ================== GARCH ==================
    if st.button("Run GARCH Volatility Forecast"):
        returns = 100 * series.pct_change().dropna()
        model = arch_model(returns, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")

        forecast = res.forecast(horizon=90)
        vol = forecast.variance.values[-1]

        fig = px.line(y=vol, title="Forecasted Volatility")
        st.plotly_chart(fig, use_container_width=True)

    # ================== LSTM ==================
    if st.button("Run LSTM Forecast"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1))

        X, y = [], []
        look_back = 30
        for i in range(len(scaled) - look_back):
            X.append(scaled[i:i + look_back])
            y.append(scaled[i + look_back])

        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, input_shape=(look_back, 1)),
            Dense(1)
        ])
        model.compile(loss="mse", optimizer="adam")
        model.fit(X[:-90], y[:-90], epochs=10, batch_size=16, verbose=0)

        preds = scaler.inverse_transform(model.predict(X[-90:]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual"))
        fig.add_trace(go.Scatter(x=test.index, y=preds.flatten(), name="LSTM Forecast"))
        fig.update_layout(title="LSTM Forecast vs Actual")
        st.plotly_chart(fig, use_container_width=True)
