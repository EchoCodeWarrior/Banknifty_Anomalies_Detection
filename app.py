import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# --- App Title and Description ---
st.set_page_config(layout="wide")
st.title("30-Year Bank Nifty Momentum and Anomaly Detection")
st.write("""
This application analyzes 30 years of historical Bank Nifty data to identify market momentum and detect significant anomalies.
The core of the analysis uses an **Isolation Forest** model to find data points that are statistically different from the norm, which often correspond to major market events.
Use the controls in the sidebar to customize the analysis and explore the data.
""")

# --- Major Event Annotations ---
# This dictionary adds context to the identified anomalies.
MAJOR_EVENTS = {
    '2008-10-24': 'Global Financial Crisis Peak Fear',
    '2016-11-09': 'Indian Banknote Demonetization',
    '2020-03-23': 'COVID-19 Pandemic Market Bottom',
    '2021-02-01': 'Union Budget 2021 Rally',
    # You can research and add more significant dates from your anomaly table.
}


# --- Load and Cache Data ---
@st.cache_data
def load_data(file_path):
    """Loads, cleans, and processes the market data."""
    try:
        df = pd.read_csv(file_path)
        # Standardize column names
        df.rename(columns={'DateTime': 'Date', 'NIFTY BANK': 'Close', 'Volume': 'Volume'}, inplace=True)

        if 'Volume' not in df.columns:
            df['Volume'] = 0

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date', 'Close'], inplace=True)
        df.set_index('Date', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found. Please ensure 'chart.csv' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Feature Engineering and Model Training ---
@st.cache_data
def train_anomaly_model(_df, contamination=0.005):
    """Engineers features and trains the anomaly detection model."""
    df = _df.copy()
    df['price_change_pct'] = df['Close'].pct_change().fillna(0) * 100
    df['volatility_21d'] = df['price_change_pct'].rolling(window=21).std().fillna(0)
    df['volume_change_pct'] = df['Volume'].pct_change().fillna(0) * 100

    features = ['price_change_pct', 'volatility_21d', 'volume_change_pct']
    X = df[features]

    model = IsolationForest(n_estimators=100, contamination=float(contamination), random_state=42)
    model.fit(X)

    df['anomaly_score'] = model.decision_function(X)
    df['is_anomaly'] = model.predict(X)
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: True if x == -1 else False)
    return df

# --- Main App Logic ---
file_path = 'chart.csv'
df_original = load_data(file_path)

if df_original is not None:
    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Analysis Controls")
    
    st.sidebar.markdown("**Date Range Selection**")
    start_date = st.sidebar.date_input("Start Date", df_original.index.min().date())
    end_date = st.sidebar.date_input("End Date", df_original.index.max().date())

    st.sidebar.markdown("**Anomaly Detection**")
    contamination_level = st.sidebar.slider(
        "Anomaly Sensitivity",
        min_value=0.001, max_value=0.05, value=0.005, step=0.001,
        help="Higher values will flag more data points as anomalies."
    )
    show_anomalies = st.sidebar.checkbox("Highlight Anomalies", value=True)

    st.sidebar.markdown("**Technical Indicators**")
    show_ma_50 = st.sidebar.checkbox("Show 50-Day Moving Average", value=True)
    show_ma_200 = st.sidebar.checkbox("Show 200-Day Moving Average", value=False)
    show_bollinger_bands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    
    # --- Data Processing ---
    df_analyzed = train_anomaly_model(df_original, contamination=contamination_level)
    
    # Filter data based on date range
    df_filtered = df_analyzed.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

    # Calculate indicators on the filtered data
    if show_ma_50:
        df_filtered['MA50'] = df_filtered['Close'].rolling(window=50).mean()
    if show_ma_200:
        df_filtered['MA200'] = df_filtered['Close'].rolling(window=200).mean()
    if show_bollinger_bands:
        window_bb = 20
        df_filtered['BB_Middle'] = df_filtered['Close'].rolling(window=window_bb).mean()
        df_filtered['BB_Std'] = df_filtered['Close'].rolling(window=window_bb).std()
        df_filtered['BB_Upper'] = df_filtered['BB_Middle'] + (df_filtered['BB_Std'] * 2)
        df_filtered['BB_Lower'] = df_filtered['BB_Middle'] - (df_filtered['BB_Std'] * 2)

    # --- Charting ---
    st.header("Market Price and Momentum Analysis")
    fig = go.Figure()

    # Base Price Line
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Close'], mode='lines', name='Close Price', line=dict(color='skyblue', width=2)))

    # Technical Indicators
    if show_ma_50:
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA50'], mode='lines', name='50-Day MA', line=dict(color='orange', width=1.5, dash='dot')))
    if show_ma_200:
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA200'], mode='lines', name='200-Day MA', line=dict(color='purple', width=1.5, dash='dot')))
    if show_bollinger_bands:
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['BB_Upper'], mode='lines', name='Upper Band', line=dict(color='gray', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['BB_Lower'], mode='lines', name='Lower Band', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))

    # Anomalies
    if show_anomalies:
        anomalies_df = df_filtered[df_filtered['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalies_df.index, y=anomalies_df['Close'],
            mode='markers', name='Detected Anomaly',
            marker=dict(color='red', size=8, symbol='circle', line=dict(color='black', width=1))
        ))
    
    # Event Annotations
    for date_str, event in MAJOR_EVENTS.items():
        event_date = pd.to_datetime(date_str)
        if event_date in df_filtered.index:
            fig.add_annotation(
                x=event_date, y=df_filtered.loc[event_date]['Close'], text=event,
                showarrow=True, arrowhead=2, ax=0, ay=-40,
                bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ff7f0e", opacity=0.8
            )

    fig.update_layout(
        title_text="Bank Nifty Price History with Technical Indicators and Anomaly Detection",
        xaxis_title="Date", yaxis_title="Price", legend_title="Indicators",
        xaxis_rangeslider_visible=True, height=650
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Anomaly Table ---
    if show_anomalies:
        st.header("Detected Anomalies")
        st.write("The table below lists the dates flagged as anomalous by the model, based on unusual daily price and volume changes, and volatility. Sort by `anomaly_score` to see the most significant events.")
        
        # Add event descriptions to the table
        anomalies_df['Event'] = anomalies_df.index.strftime('%Y-%m-%d').map(MAJOR_EVENTS).fillna('N/A')
        
        display_cols = ['Event', 'Close', 'Volume', 'price_change_pct', 'volatility_21d', 'volume_change_pct', 'anomaly_score']
        st.dataframe(anomalies_df[display_cols].sort_values('anomaly_score'))

    # --- Data Summary Expander ---
    with st.expander("Show Raw Data Summary"):
        st.write("Descriptive Statistics for the selected period:")
        st.write(df_filtered[['Close', 'Volume', 'price_change_pct']].describe())
