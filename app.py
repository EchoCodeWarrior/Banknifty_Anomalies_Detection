import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# --- App Title and Description ---
st.set_page_config(layout="wide")
st.title("30-Year Market Momentum and Anomaly Detection")
st.write("""
This application analyzes 30 years of historical market data to identify momentum and detect anomalies.
An **Isolation Forest** model is used to find data points that are statistically different from the norm, which could indicate unusual market events.
""")

# --- Load and Cache Data ---
@st.cache_data
def load_data(file_path):
    """Loads, cleans, and processes the market data."""
    try:
        df = pd.read_csv(file_path)
        # Rename columns to the expected format for the rest of the script
        df.rename(columns={'DateTime': 'Date', 'NIFTY BANK': 'Close'}, inplace=True)

        # Add a placeholder 'Volume' column if it doesn't exist
        if 'Volume' not in df.columns:
            df['Volume'] = 0

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date', 'Close'], inplace=True)
        df.set_index('Date', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found in the repository. Please make sure it has been uploaded to GitHub.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Feature Engineering and Model Training ---
@st.cache_data
def train_anomaly_model(_df):
    """Engineers features and trains the anomaly detection model."""
    df = _df.copy()
    df['price_change_pct'] = df['Close'].pct_change().fillna(0) * 100
    df['volatility_21d'] = df['price_change_pct'].rolling(window=21).std().fillna(0)
    features = ['price_change_pct', 'volatility_21d']
    X = df[features]
    model = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
    model.fit(X)
    df['anomaly_score'] = model.decision_function(X)
    df['is_anomaly'] = model.predict(X)
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: True if x == -1 else False)
    return df

# --- Main App Logic ---
# Use the relative file path for cloud deployment
file_path = 'chart.csv'
df_original = load_data(file_path)

if df_original is not None:
    st.sidebar.header("Chart Controls")
    df_analyzed = train_anomaly_model(df_original)

    show_anomalies = st.sidebar.checkbox("Highlight Anomalies", value=True)
    show_ma_50 = st.sidebar.checkbox("Show 50-Day Moving Average", value=True)
    show_ma_200 = st.sidebar.checkbox("Show 200-Day Moving Average", value=False)

    if show_ma_50:
        df_analyzed['MA50'] = df_analyzed['Close'].rolling(window=50).mean()
    if show_ma_200:
        df_analyzed['MA200'] = df_analyzed['Close'].rolling(window=200).mean()

    st.header("Market Price and Momentum Analysis")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_analyzed.index, y=df_analyzed['Close'], mode='lines', name='Close Price', line=dict(color='skyblue', width=2)))

    if show_ma_50:
        fig.add_trace(go.Scatter(x=df_analyzed.index, y=df_analyzed['MA50'], mode='lines', name='50-Day MA', line=dict(color='orange', width=1.5, dash='dot')))
    if show_ma_200:
        fig.add_trace(go.Scatter(x=df_analyzed.index, y=df_analyzed['MA200'], mode='lines', name='200-Day MA', line=dict(color='purple', width=1.5, dash='dot')))

    if show_anomalies:
        anomalies_df = df_analyzed[df_analyzed['is_anomaly']]
        fig.add_trace(go.Scatter(x=anomalies_df.index, y=anomalies_df['Close'], mode='markers', name='Detected Anomaly', marker=dict(color='red', size=8, symbol='circle', line=dict(color='black', width=1))))

    fig.update_layout(
        title_text="30-Year Price History with Momentum Indicators and Anomaly Detection",
        xaxis_title="Date", yaxis_title="Price", legend_title="Indicators",
        xaxis_rangeslider_visible=True, height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_anomalies:
        st.header("Detected Anomalies")
        st.write("The table below lists the dates flagged as anomalous by the model, based on unusual daily price changes and volatility.")
        display_cols = ['Close', 'Volume', 'price_change_pct', 'volatility_21d', 'anomaly_score']
        st.dataframe(anomalies_df[display_cols].sort_values('anomaly_score'))