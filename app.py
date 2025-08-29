import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import plotly.express as px
from datetime import datetime, timedelta

# --- Premium Page Configuration ---
st.set_page_config(
    page_title="Bank Nifty Intelligence Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Styling ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #2d3748 0%, #4a5568 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4fd1c7;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        border-right: 2px solid rgba(79, 209, 199, 0.2);
    }
    
    .css-1d391kg .css-17eq0hr {
        color: #4fd1c7;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #4fd1c7 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Custom Alert Boxes */
    .alert-success {
        background: linear-gradient(90deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 178, 172, 0.1) 100%);
        border-left: 4px solid #48bb78;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: linear-gradient(90deg, rgba(79, 209, 199, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%);
        border-left: 4px solid #4fd1c7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #4fd1c7 0%, #667eea 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 209, 199, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 209, 199, 0.4);
    }
    
    /* DataFrame Styling */
    .dataframe {
        background: rgba(45, 55, 72, 0.8);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(79, 209, 199, 0.3);
        border-top: 4px solid #4fd1c7;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Premium Header ---
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🏦 Bank Nifty Intelligence Hub</h1>
    <p class="main-subtitle">Advanced AI-Powered Market Anomaly Detection & Risk Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# --- Major Event Annotations (Enhanced) ---
MAJOR_EVENTS = {
    '2008-10-24': '🔴 Global Financial Crisis Peak Fear',
    '2009-03-09': '🟢 Financial Crisis Recovery Begin',
    '2016-11-09': '🟠 Indian Banknote Demonetization',
    '2018-10-02': '🔴 IL&FS Crisis & NBFC Concerns',
    '2020-03-23': '🔴 COVID-19 Pandemic Market Bottom',
    '2020-04-01': '🟢 COVID Recovery Rally Begins',
    '2021-02-01': '🟢 Union Budget 2021 Banking Rally',
    '2022-04-29': '🔴 RBI Rate Hike Concerns',
    '2023-01-24': '🔴 Adani Group Crisis Impact',
}

# --- Enhanced Data Loading with Progress ---
@st.cache_data
def load_data(file_path):
    """Loads, cleans, and processes the market data with enhanced error handling."""
    progress_bar = st.progress(0, text="Loading market data...")
    
    try:
        # Read the CSV file
        progress_bar.progress(25, text="Reading CSV file...")
        df = pd.read_csv(file_path)
        
        # Standardize column names
        progress_bar.progress(50, text="Processing data structure...")
        column_mapping = {
            'DateTime': 'Date', 
            'NIFTY BANK': 'Close', 
            'Volume': 'Volume',
            'Date': 'Date',
            'Close': 'Close'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Handle missing Volume column
        if 'Volume' not in df.columns:
            df['Volume'] = 100000  # Default volume for calculation purposes
        
        progress_bar.progress(75, text="Cleaning and validating data...")
        
        # Date processing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date', 'Close'], inplace=True)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        progress_bar.progress(100, text="Data loaded successfully!")
        progress_bar.empty()
        
        return df
        
    except FileNotFoundError:
        st.error("📄 **Data File Not Found**")
        st.markdown("""
        <div class="alert-info">
            <strong>Expected File:</strong> 'chart.csv' in the same directory<br>
            <strong>Required Columns:</strong> Date/DateTime, Close/NIFTY BANK, Volume (optional)
        </div>
        """, unsafe_allow_html=True)
        return None
        
    except Exception as e:
        st.error(f"⚠️ **Data Loading Error:** {str(e)}")
        return None

# --- Enhanced Feature Engineering and Model Training ---
@st.cache_data
def train_anomaly_model(_df, contamination=0.005):
    """Enhanced feature engineering and anomaly detection model."""
    progress_bar = st.progress(0, text="Training anomaly detection model...")
    
    df = _df.copy()
    
    # Progress: Feature Engineering
    progress_bar.progress(20, text="Engineering features...")
    
    # Basic features
    df['price_change_pct'] = df['Close'].pct_change().fillna(0) * 100
    df['volatility_21d'] = df['price_change_pct'].rolling(window=21).std().fillna(0)
    df['volume_change_pct'] = df['Volume'].pct_change().fillna(0) * 100
    
    # Advanced features
    df['price_momentum_5d'] = df['Close'].pct_change(periods=5).fillna(0) * 100
    df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['high_low_range'] = ((df['Close'] - df['Close'].rolling(window=5).min()) / 
                           (df['Close'].rolling(window=5).max() - df['Close'].rolling(window=5).min())).fillna(0.5)
    
    progress_bar.progress(50, text="Preparing model features...")
    
    # Feature selection
    features = ['price_change_pct', 'volatility_21d', 'volume_change_pct', 
               'price_momentum_5d', 'volume_ma_ratio', 'high_low_range']
    X = df[features].fillna(0)
    
    progress_bar.progress(75, text="Training Isolation Forest model...")
    
    # Train model
    model = IsolationForest(
        n_estimators=200, 
        contamination=float(contamination), 
        random_state=42,
        max_features=1.0
    )
    model.fit(X)
    
    # Generate predictions
    df['anomaly_score'] = model.decision_function(X)
    df['is_anomaly'] = model.predict(X) == -1
    
    progress_bar.progress(100, text="Model training completed!")
    progress_bar.empty()
    
    return df

# --- Enhanced Sidebar with Premium Design ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #4fd1c7 0%, #667eea 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">⚙️ Control Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📅 **Analysis Period**")
    
    # Data loading
    file_path = 'chart.csv'
    df_original = load_data(file_path)
    
    if df_original is not None:
        # Date range controls
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From", 
                df_original.index.min().date(),
                help="Select start date for analysis"
            )
        with col2:
            end_date = st.date_input(
                "To", 
                df_original.index.max().date(),
                help="Select end date for analysis"
            )
        
        st.markdown("### 🎯 **Anomaly Detection**")
        contamination_level = st.slider(
            "Detection Sensitivity",
            min_value=0.001, 
            max_value=0.02, 
            value=0.005, 
            step=0.001,
            help="Higher values detect more anomalies",
            format="%.3f"
        )
        
        show_anomalies = st.toggle("🔍 Highlight Anomalies", value=True)
        
        st.markdown("### 📊 **Technical Indicators**")
        show_ma_50 = st.toggle("50-Day Moving Average", value=True)
        show_ma_200 = st.toggle("200-Day Moving Average", value=False)
        show_bollinger_bands = st.toggle("Bollinger Bands", value=True)
        show_volume = st.toggle("Volume Analysis", value=False)
        
        st.markdown("### 🎨 **Chart Appearance**")
        chart_theme = st.selectbox(
            "Color Theme",
            ["Professional Dark", "Ocean Blue", "Forest Green", "Sunset Orange"],
            index=0
        )
        
        # Theme color mapping
        theme_colors = {
            "Professional Dark": {"primary": "#4fd1c7", "secondary": "#667eea"},
            "Ocean Blue": {"primary": "#0077be", "secondary": "#00a8cc"},
            "Forest Green": {"primary": "#2d5a27", "secondary": "#40826d"},
            "Sunset Orange": {"primary": "#ff6b35", "secondary": "#f7931e"}
        }
        
        selected_theme = theme_colors[chart_theme]

# --- Main Application Logic ---
if df_original is not None:
    # Train model and analyze data
    with st.spinner("🤖 AI Model Processing..."):
        df_analyzed = train_anomaly_model(df_original, contamination=contamination_level)
    
    # Filter data based on date range
    df_filtered = df_analyzed.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    
    # Calculate technical indicators
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
    
    # --- Key Metrics Dashboard ---
    st.markdown('<h2 class="section-header">📈 Market Intelligence Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_anomalies = df_filtered['is_anomaly'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_anomalies}</div>
            <div class="metric-label">Anomalies Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        current_price = df_filtered['Close'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{current_price:,.0f}</div>
            <div class="metric-label">Latest Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_drawdown = ((df_filtered['Close'] / df_filtered['Close'].expanding().max()) - 1).min() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max_drawdown:.1f}%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_volatility = df_filtered['volatility_21d'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_volatility:.2f}%</div>
            <div class="metric-label">Avg Volatility</div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Enhanced Main Chart ---
    st.markdown('<h2 class="section-header">💹 Advanced Price Analysis</h2>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Base price line with enhanced styling
    fig.add_trace(go.Scatter(
        x=df_filtered.index, 
        y=df_filtered['Close'], 
        mode='lines', 
        name='Bank Nifty',
        line=dict(color=selected_theme["primary"], width=3),
        hovertemplate="<b>%{x}</b><br>Price: ₹%{y:,.0f}<extra></extra>"
    ))
    
    # Technical indicators with premium styling
    if show_ma_50:
        fig.add_trace(go.Scatter(
            x=df_filtered.index, 
            y=df_filtered['MA50'], 
            mode='lines', 
            name='MA 50',
            line=dict(color='#ffa500', width=2, dash='dot'),
            hovertemplate="<b>50-Day MA</b><br>%{x}<br>₹%{y:,.0f}<extra></extra>"
        ))
    
    if show_ma_200:
        fig.add_trace(go.Scatter(
            x=df_filtered.index, 
            y=df_filtered['MA200'], 
            mode='lines', 
            name='MA 200',
            line=dict(color='#9370db', width=2, dash='dot'),
            hovertemplate="<b>200-Day MA</b><br>%{x}<br>₹%{y:,.0f}<extra></extra>"
        ))
    
    if show_bollinger_bands:
        # Upper band
        fig.add_trace(go.Scatter(
            x=df_filtered.index, 
            y=df_filtered['BB_Upper'], 
            mode='lines', 
            name='Upper BB',
            line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
            hovertemplate="<b>Upper Bollinger Band</b><br>%{x}<br>₹%{y:,.0f}<extra></extra>"
        ))
        
        # Lower band with fill
        fig.add_trace(go.Scatter(
            x=df_filtered.index, 
            y=df_filtered['BB_Lower'], 
            mode='lines', 
            name='Lower BB',
            line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
            fill='tonexty', 
            fillcolor='rgba(128,128,128,0.1)',
            hovertemplate="<b>Lower Bollinger Band</b><br>%{x}<br>₹%{y:,.0f}<extra></extra>"
        ))
    
    # Enhanced anomaly visualization
    if show_anomalies:
        anomalies_df = df_filtered[df_filtered['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalies_df.index, 
            y=anomalies_df['Close'],
            mode='markers', 
            name='Market Anomalies',
            marker=dict(
                color='#ff4757', 
                size=12, 
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            hovertemplate="<b>⚠️ ANOMALY DETECTED</b><br>%{x}<br>Price: ₹%{y:,.0f}<br>Change: %{customdata:.2f}%<extra></extra>",
            customdata=anomalies_df['price_change_pct']
        ))
    
    # Enhanced event annotations
    for date_str, event in MAJOR_EVENTS.items():
        event_date = pd.to_datetime(date_str)
        if event_date in df_filtered.index:
            fig.add_annotation(
                x=event_date, 
                y=df_filtered.loc[event_date]['Close'], 
                text=event,
                showarrow=True, 
                arrowhead=2, 
                ax=0, 
                ay=-50,
                bordercolor="#2d3748", 
                borderwidth=2, 
                borderpad=8, 
                bgcolor="rgba(79, 209, 199, 0.9)", 
                font=dict(color="white", size=11, family="Inter"),
                opacity=0.95,
                arrowcolor=selected_theme["secondary"]
            )
    
    # Premium chart layout
    fig.update_layout(
        title={
            'text': "🏦 Bank Nifty Advanced Technical Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#4fd1c7', 'family': 'Inter'}
        },
        xaxis_title="Timeline", 
        yaxis_title="Price (₹)", 
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(45, 55, 72, 0.8)",
            bordercolor="rgba(79, 209, 199, 0.5)",
            borderwidth=1
        ),
        xaxis_rangeslider_visible=True,
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Volume Analysis (if enabled) ---
    if show_volume:
        st.markdown('<h2 class="section-header">📊 Volume Analysis</h2>', unsafe_allow_html=True)
        
        fig_vol = go.Figure()
        
        # Volume bars with color coding
        colors = ['#ff4757' if anomaly else selected_theme["primary"] for anomaly in df_filtered['is_anomaly']]
        
        fig_vol.add_trace(go.Bar(
            x=df_filtered.index,
            y=df_filtered['Volume'],
            name='Volume',
            marker_color=colors,
            hovertemplate="<b>Volume Analysis</b><br>%{x}<br>Volume: %{y:,.0f}<extra></extra>"
        ))
        
        fig_vol.update_layout(
            title={
                'text': "📈 Trading Volume with Anomaly Highlighting",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#4fd1c7', 'family': 'Inter'}
            },
            xaxis_title="Date",
            yaxis_title="Volume",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # --- Enhanced Anomaly Analysis Table ---
    if show_anomalies and total_anomalies > 0:
        st.markdown('<h2 class="section-header">🔍 Detailed Anomaly Analysis</h2>', unsafe_allow_html=True)
        
        # Prepare enhanced anomaly data
        anomalies_display = df_filtered[df_filtered['is_anomaly']].copy()
        
        # Add event descriptions
        anomalies_display['Event'] = anomalies_display.index.strftime('%Y-%m-%d').map(MAJOR_EVENTS).fillna('Market Event')
        
        # Add severity classification
        def classify_severity(score):
            if score <= -0.5:
                return "🔴 Extreme"
            elif score <= -0.3:
                return "🟠 High"
            elif score <= -0.1:
                return "🟡 Moderate"
            else:
                return "🟢 Low"
        
        anomalies_display['Severity'] = anomalies_display['anomaly_score'].apply(classify_severity)
        
        # Format the display columns
        display_data = anomalies_display[[
            'Event', 'Severity', 'Close', 'price_change_pct', 
            'volatility_21d', 'volume_change_pct', 'anomaly_score'
        ]].copy()
        
        # Round numerical values
        display_data['Close'] = display_data['Close'].round(0)
        display_data['price_change_pct'] = display_data['price_change_pct'].round(2)
        display_data['volatility_21d'] = display_data['volatility_21d'].round(2)
        display_data['volume_change_pct'] = display_data['volume_change_pct'].round(2)
        display_data['anomaly_score'] = display_data['anomaly_score'].round(3)
        
        # Rename columns for display
        display_data.columns = [
            'Market Event', 'Risk Level', 'Price (₹)', 'Daily Change (%)', 
            'Volatility (21D)', 'Volume Change (%)', 'Anomaly Score'
        ]
        
        # Sort by anomaly score (most anomalous first)
        display_data = display_data.sort_values('Anomaly Score')
        
        st.dataframe(
            display_data, 
            use_container_width=True,
            height=400
        )
        
        # Anomaly insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="alert-info">
                <strong>🎯 Anomaly Detection Insights</strong><br>
                • Lower anomaly scores indicate higher deviation from normal patterns<br>
                • Red markers on chart represent detected anomalies<br>
                • Historical events provide context for market behavior
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            most_anomalous = display_data.iloc[0]
            st.markdown(f"""
            <div class="alert-success">
                <strong>🏆 Most Significant Anomaly</strong><br>
                <strong>Event:</strong> {most_anomalous['Market Event']}<br>
                <strong>Risk Level:</strong> {most_anomalous['Risk Level']}<br>
                <strong>Score:</strong> {most_anomalous['Anomaly Score']}
            </div>
            """, unsafe_allow_html=True)
    
    # --- Enhanced Data Summary ---
    with st.expander("📊 **Comprehensive Data Analytics**", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 **Price Statistics**")
            price_stats = df_filtered[['Close', 'price_change_pct']].describe().round(2)
            st.dataframe(price_stats, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 **Volume & Volatility**")
            vol_stats = df_filtered[['Volume', 'volatility_21d']].describe().round(2)
            st.dataframe(vol_stats, use_container_width=True)
        
        # Risk metrics
        st.markdown("#### ⚠️ **Risk Metrics**")
        risk_metrics = {
            'Sharpe Ratio (Approx)': f"{(df_filtered['price_change_pct'].mean() / df_filtered['price_change_pct'].std() * (252**0.5)):.2f}",
            'Maximum Single Day Loss': f"{df_filtered['price_change_pct'].min():.2f}%",
            'Maximum Single Day Gain': f"{df_filtered['price_change_pct'].max():.2f}
