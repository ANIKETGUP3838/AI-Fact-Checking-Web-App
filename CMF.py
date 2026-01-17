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
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide")
st.title("📈 Cryptocurrency Market Forecasting App")

tab1, tab2 = st.tabs(["📄 Project Summary", "📊 Forecasting App"])

# ================= HELPERS =================
def find_first_existing_column(data, candidates):
    for col in candidates:
        if col in data.columns:
            return col
    return None

# ================= TAB 1 =================
with tab1:
    st.markdown("""
    **Cryptocurrency Market Forecasting**
    
    Models used:
    - ARIMA
    - SARIMA
    - Exponential Smoothing
    - GARCH
    - LSTM
    """)

# ================= TAB 2 =================
with tab2:

    uploaded_file = st.sidebar.file_uploader(
        "Upload crypto-markets.csv",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file to continue.")
        st.stop()

    # ---------- READ CSV (DO NOT RENAME THIS VARIABLE) ----------
    crypto_df = pd.read_csv(uploaded_file)

    if not isinstance(crypto_df, pd.DataFrame):
        st.error("Uploaded file is not a valid CSV.")
        st.stop()

    # ---------- DETECT DATE COLUMN ----------
    date_col = find_first_existing_column(
        crypto_df,
        ["date", "timestamp", "time", "datetime"]
    )

    if date_col is None:
        st.error("No date column found.")
        st.write(list(crypto_df.columns))
        st.stop()

    crypto_df[date_col] = pd.to_datetime(
        crypto_df[date_col],
        errors="coerce"
    )

    crypto_df.dropna(subset=[date_col], inplace=True)
    crypto_df.sort_values(date_col, inplace=True)

    # ---------- DETECT CRYPTO COLUMN ----------
    crypto_col = find_first_existing_column(
        crypto_df,
        ["symbol", "coin", "name", "id", "asset"]
    )

    if crypto_col is None:
        st.error("No cryptocurrency identifier column found.")
        st.write(list(crypto_df.columns))
        st.stop()

    # ---------- SIDEBAR FILTERS ----------
    st.sidebar.success(f"Crypto column: {crypto_col}")
    st.sidebar.success(f"Date column: {date_col}")

    crypto_value = st.sidebar.selectbox(
        "Cryptocurrency",
        sorted(crypto_df[crypto_col].astype(str).unique())
    )

    filtered_df = crypto_df[
        crypto_df[crypto_col].astype(str) == crypto_value
    ].copy()

    filtered_df.set_index(date_col, inplace=True)

    # ---------- NUMERIC TARGET ----------
    numeric_cols = filtered_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns available.")
        st.stop()

    target_col = st.sidebar.selectbox(
        "Target Variable",
        numeric_cols
    )

    series = filtered_df[target_col].dropna()

    if len(series) < 120:
        st.warning("Not enough data points (minimum 120).")
        st.stop()

    # ---------- PREVIEW ----------
    st.subheader(f"{crypto_value} — {target_col.upper()}")
    st.write(series.head())

    # ---------- TIME SERIES ----------
    st.plotly_chart(
        px.line(series, title="Time Series"),
        use_container_width=True
    )

    # ---------- DECOMPOSITION ----------
    decomposition = sm.tsa.seasonal_decompose(
        series,
        model="additive",
        period=30
    )

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"]
    )

    fig.add_trace(go.Scatter(y=decomposition.observed), 1, 1)
    fig.add_trace(go.Scatter(y=decomposition.trend), 2, 1)
    fig.add_trace(go.Scatter(y=decomposition.seasonal), 3, 1)
    fig.add_trace(go.Scatter(y=decomposition.resid), 4, 1)

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- ADF ----------
    st.subheader("ADF Test")
    adf = adfuller(series)
    st.write({
        "ADF Statistic": adf[0],
        "p-value": adf[1]
    })

    # ---------- TRAIN / TEST ----------
    train = series[:-90]
    test = series[-90:]

    # ---------- ARIMA ----------
    if st.button("Run ARIMA"):
        model = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = model.forecast(90)
        st.write("RMSE:", np.sqrt(mean_squared_error(test, forecast)))

    # ---------- SARIMA ----------
    if st.button("Run SARIMA"):
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7)
        ).fit()

        forecast = model.get_forecast(90).predicted_mean
        st.write("RMSE:", np.sqrt(mean_squared_error(test, forecast)))

    # ---------- EXP SMOOTH ----------
    if st.button("Run Exponential Smoothing"):
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=30
        ).fit()

        forecast = model.forecast(90)
        st.write("RMSE:", np.sqrt(mean_squared_error(test, forecast)))

    # ---------- GARCH ----------
    if st.button("Run GARCH"):
        returns = 100 * series.pct_change().dropna()
        model = arch_model(returns, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        vol = res.forecast(horizon=90).variance.values[-1]
        st.plotly_chart(px.line(y=vol), use_container_width=True)

    # ---------- LSTM ----------
    if st.button("Run LSTM"):
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
        st.plotly_chart(fig, use_container_width=True)
