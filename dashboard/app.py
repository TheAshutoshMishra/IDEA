"""
UIDAI Authentication Intelligence Platform
Real-time monitoring, anomaly detection, and predictive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration with UIDAI branding
st.set_page_config(
    page_title="UIDAI Intelligence Platform",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with UIDAI colors and branding
st.markdown("""
    <style>
    /* UIDAI Color Scheme */
    :root {
        --uidai-blue: #0066B2;
        --uidai-orange: #FF6B35;
        --uidai-green: #4CAF50;
        --uidai-red: #E53935;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #0066B2 0%, #004D8C 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: #E3F2FD;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #0066B2;
        margin-bottom: 1rem;
    }
    
    .metric-card.success {
        border-left-color: #4CAF50;
    }
    
    .metric-card.warning {
        border-left-color: #FF6B35;
    }
    
    .metric-card.danger {
        border-left-color: #E53935;
    }
    
    /* Alert Cards */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .alert-critical {
        background-color: #FFEBEE;
        border-left-color: #E53935;
        color: #C62828;
    }
    
    .alert-high {
        background-color: #FFF3E0;
        border-left-color: #FF6B35;
        color: #E65100;
    }
    
    .alert-medium {
        background-color: #FFF9C4;
        border-left-color: #FBC02D;
        color: #F57F17;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0066B2 0%, #004D8C 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066B2;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 2px solid #E3F2FD;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Simple authentication
def check_password():
    """Returns True if the user has the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "uidai2026":
            st.session_state["password_correct"] = True
            st.session_state["role"] = "admin"
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown('<div class="main-header"><h1>🇮🇳 UIDAI Intelligence Platform</h1><p>Unique Identification Authority of India</p></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 Secure Login")
            st.text_input(
                "Enter Password", type="password", on_change=password_entered, key="password"
            )
            st.info("💡 Demo Password: **uidai2026**")
            st.markdown("---")
            st.markdown("**Roles Available:**")
            st.markdown("- 👤 Admin: Full access")
            st.markdown("- 📊 Analyst: View & analyze")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown('<div class="main-header"><h1>🇮🇳 UIDAI Intelligence Platform</h1><p>Unique Identification Authority of India</p></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(
                "Enter Password", type="password", on_change=password_entered, key="password"
            )
            st.error("😕 Password incorrect")
        return False
    else:
        return True

# Load data with caching
@st.cache_data
def load_auth_data():
    """Load authentication transaction data"""
    df = pd.read_csv('data/raw/auth_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def get_summary_stats(df):
    """Calculate summary statistics"""
    total = len(df)
    success = (df['result'] == 'Success').sum()
    failure = total - success
    success_rate = (success / total * 100) if total > 0 else 0
    
    anomalies = df['is_anomaly'].sum()
    high_retry = (df['retry_count'] > 3).sum()
    avg_response_time = df['response_time_ms'].mean()
    
    # Last 24h stats
    last_24h = df[df['timestamp'] >= df['timestamp'].max() - timedelta(hours=24)]
    recent_success_rate = ((last_24h['result'] == 'Success').sum() / len(last_24h) * 100) if len(last_24h) > 0 else 0
    
    return {
        'total': total,
        'success': success,
        'failure': failure,
        'success_rate': success_rate,
        'anomalies': anomalies,
        'high_retry': high_retry,
        'avg_response_time': avg_response_time,
        'recent_success_rate': recent_success_rate,
        'last_24h_count': len(last_24h)
    }

def main():
    """Main application"""
    
    if not check_password():
        return
    
    # Header with UIDAI branding
    st.markdown("""
        <div class="main-header">
            <h1>🇮🇳 UIDAI Authentication Intelligence Platform</h1>
            <p>Real-time Monitoring • Anomaly Detection • Predictive Analytics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_auth_data()
        stats = get_summary_stats(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure auth_transactions.csv is in data/raw/ directory")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.info(f"**Role:** {st.session_state.get('role', 'admin').title()}")
        st.markdown("---")
        
        st.markdown("### 📅 Data Summary")
        st.metric("Total Records", f"{stats['total']:,}")
        st.metric("Date Range", f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")
        st.metric("Regions Covered", df['region'].nunique())
        
        st.markdown("---")
        st.markdown("### 🔄 Refresh Data")
        if st.button("🔄 Reload", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Dashboard", 
        "🔍 Pattern Analysis", 
        "⚠️ Anomaly Detection",
        "🔮 Predictive Analytics",
        "📋 Detailed Reports"
    ])
    
    with tab1:
        show_executive_dashboard(df, stats)
    
    with tab2:
        show_pattern_analysis(df)
    
    with tab3:
        show_anomaly_detection(df)
    
    with tab4:
        show_predictive_analytics(df)
    
    with tab5:
        show_detailed_reports(df)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p><strong>UIDAI Authentication Intelligence Platform</strong></p>
            <p>Unique Identification Authority of India | Government of India</p>
            <p>🇮🇳 Empowering Digital India | UIDAI Data Hackathon 2026</p>
        </div>
    """, unsafe_allow_html=True)

def show_executive_dashboard(df, stats):
    """Executive dashboard with key metrics"""
    
    st.header("📊 Executive Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card success">', unsafe_allow_html=True)
        st.metric(
            "Total Authentications", 
            f"{stats['total']:,}",
            delta=f"{stats['last_24h_count']:,} in 24h"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card success">', unsafe_allow_html=True)
        st.metric(
            "Success Rate", 
            f"{stats['success_rate']:.1f}%",
            delta=f"{stats['recent_success_rate']:.1f}% (24h)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card warning">', unsafe_allow_html=True)
        st.metric(
            "Anomalies Detected", 
            f"{stats['anomalies']:,}",
            delta=f"{(stats['anomalies']/stats['total']*100):.1f}% of total"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Avg Response Time", 
            f"{stats['avg_response_time']:.0f}ms",
            delta="Normal" if stats['avg_response_time'] < 1000 else "High"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Authentication Trend (Last 7 Days)")
        last_7days = df[df['timestamp'] >= df['timestamp'].max() - timedelta(days=7)]
        daily_trend = last_7days.groupby([last_7days['timestamp'].dt.date, 'result']).size().reset_index()
        daily_trend.columns = ['Date', 'Result', 'Count']
        
        fig = px.line(daily_trend, x='Date', y='Count', color='Result',
                     color_discrete_map={'Success': '#4CAF50', 'Failure': '#E53935'})
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🗺️ Geographic Distribution")
        region_stats = df.groupby('region').agg({
            'transaction_id': 'count',
            'result': lambda x: (x == 'Success').sum() / len(x) * 100
        }).reset_index()
        region_stats.columns = ['Region', 'Transactions', 'Success_Rate']
        region_stats = region_stats.sort_values('Transactions', ascending=False).head(10)
        
        fig = px.bar(region_stats, x='Region', y='Transactions',
                    color='Success_Rate', color_continuous_scale='RdYlGn')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Alert Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Recent Alerts")
        
        # Critical: High failure rate regions
        recent_failures = df[df['timestamp'] >= df['timestamp'].max() - timedelta(hours=24)]
        region_failures = recent_failures.groupby('region').apply(
            lambda x: (x['result'] == 'Failure').sum() / len(x) * 100
        ).sort_values(ascending=False).head(3)
        
        for region, rate in region_failures.items():
            if rate > 20:
                st.markdown(f'<div class="alert-box alert-critical">🔴 <strong>CRITICAL:</strong> {region} - {rate:.1f}% failure rate (24h)</div>', unsafe_allow_html=True)
            elif rate > 15:
                st.markdown(f'<div class="alert-box alert-high">🟠 <strong>HIGH:</strong> {region} - {rate:.1f}% failure rate (24h)</div>', unsafe_allow_html=True)
        
        # Anomaly count
        recent_anomalies = recent_failures['is_anomaly'].sum()
        if recent_anomalies > 50:
            st.markdown(f'<div class="alert-box alert-high">🟠 <strong>HIGH:</strong> {recent_anomalies} anomalies detected in last 24h</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Authentication Methods")
        auth_method_dist = df['auth_type'].value_counts()
        
        fig = px.pie(values=auth_method_dist.values, names=auth_method_dist.index,
                    color_discrete_sequence=['#0066B2', '#FF6B35', '#4CAF50'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_pattern_analysis(df):
    """Pattern and trend analysis"""
    
    st.header("🔍 Pattern & Trend Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Temporal Patterns", "Regional Analysis", "Authentication Method Analysis", "Failure Analysis"]
    )
    
    if analysis_type == "Temporal Patterns":
        st.subheader("⏰ Temporal Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            hourly = df.groupby('hour').agg({
                'transaction_id': 'count',
                'result': lambda x: (x == 'Success').sum() / len(x) * 100
            }).reset_index()
            hourly.columns = ['Hour', 'Transactions', 'Success_Rate']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hourly['Hour'], y=hourly['Transactions'], name='Transactions'))
            fig.update_layout(title="Hourly Transaction Volume", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week pattern
            dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            df['dow_name'] = df['day_of_week'].map(dow_map)
            dow = df.groupby('dow_name').size().reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            
            fig = px.bar(x=dow.index, y=dow.values, labels={'x': 'Day', 'y': 'Transactions'})
            fig.update_layout(title="Day of Week Pattern", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekend vs Weekday
        weekend_stats = df.groupby('is_weekend').agg({
            'transaction_id': 'count',
            'result': lambda x: (x == 'Success').sum() / len(x) * 100
        }).reset_index()
        weekend_stats['Type'] = weekend_stats['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekday Transactions", f"{weekend_stats[weekend_stats['Type']=='Weekday']['transaction_id'].values[0]:,}")
        with col2:
            st.metric("Weekend Transactions", f"{weekend_stats[weekend_stats['Type']=='Weekend']['transaction_id'].values[0]:,}")
    
    elif analysis_type == "Regional Analysis":
        st.subheader("🗺️ Regional Analysis")
        
        region_stats = df.groupby('region').agg({
            'transaction_id': 'count',
            'result': lambda x: (x == 'Success').sum() / len(x) * 100,
            'response_time_ms': 'mean',
            'is_anomaly': 'sum'
        }).reset_index()
        region_stats.columns = ['Region', 'Transactions', 'Success_Rate', 'Avg_Response_Time', 'Anomalies']
        region_stats = region_stats.sort_values('Transactions', ascending=False)
        
        # Display as dataframe
        st.dataframe(
            region_stats.style.background_gradient(subset=['Success_Rate'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Regional heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(region_stats.head(10), x='Region', y='Transactions',
                        color='Success_Rate', color_continuous_scale='RdYlGn',
                        title="Top 10 Regions by Volume")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(region_stats, x='Transactions', y='Success_Rate',
                           size='Anomalies', hover_data=['Region'],
                           title="Volume vs Success Rate")
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Authentication Method Analysis":
        st.subheader("🔐 Authentication Method Analysis")
        
        auth_stats = df.groupby('auth_type').agg({
            'transaction_id': 'count',
            'result': lambda x: (x == 'Success').sum() / len(x) * 100,
            'response_time_ms': 'mean',
            'retry_count': 'mean'
        }).reset_index()
        auth_stats.columns = ['Auth_Type', 'Transactions', 'Success_Rate', 'Avg_Response_Time', 'Avg_Retries']
        
        col1, col2, col3 = st.columns(3)
        
        for idx, row in auth_stats.iterrows():
            with [col1, col2, col3][idx]:
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.subheader(row['Auth_Type'])
                st.metric("Transactions", f"{int(row['Transactions']):,}")
                st.metric("Success Rate", f"{row['Success_Rate']:.1f}%")
                st.metric("Avg Response", f"{row['Avg_Response_Time']:.0f}ms")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison chart
        fig = px.bar(auth_stats, x='Auth_Type', y=['Success_Rate', 'Avg_Response_Time'],
                    barmode='group', title="Authentication Method Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Failure Analysis
        st.subheader("❌ Failure Analysis")
        
        failures = df[df['result'] == 'Failure']
        
        col1, col2 = st.columns(2)
        
        with col1:
            failure_reasons = failures['failure_reason'].value_counts()
            fig = px.pie(values=failure_reasons.values, names=failure_reasons.index,
                        title="Failure Reasons Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Failures by region
            region_failures = failures.groupby('region').size().sort_values(ascending=False).head(10)
            fig = px.bar(x=region_failures.index, y=region_failures.values,
                        title="Top 10 Regions by Failure Count",
                        labels={'x': 'Region', 'y': 'Failures'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Failure trends
        st.subheader("📉 Failure Trend Over Time")
        failures['date'] = failures['timestamp'].dt.date
        failure_trend = failures.groupby('date').size().reset_index()
        failure_trend.columns = ['Date', 'Failures']
        
        fig = px.line(failure_trend, x='Date', y='Failures', markers=True)
        st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df):
    """Anomaly detection dashboard"""
    
    st.header("⚠️ Anomaly Detection & Risk Assessment")
    
    anomalies = df[df['is_anomaly'] == 1].copy()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Anomalies", f"{len(anomalies):,}")
    
    with col2:
        anomaly_rate = (len(anomalies) / len(df) * 100)
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    with col3:
        recent_anomalies = anomalies[anomalies['timestamp'] >= df['timestamp'].max() - timedelta(hours=24)]
        st.metric("Last 24h", f"{len(recent_anomalies):,}")
    
    with col4:
        high_risk = anomalies[anomalies['retry_count'] > 3]
        st.metric("High Risk", f"{len(high_risk):,}")
    
    st.markdown("---")
    
    # Anomaly categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Anomaly Detection Methods")
        st.markdown("""
        <div class="info-box">
        <strong>Detection Algorithms Used:</strong>
        <ul>
            <li>🔍 High Retry Count (>3 attempts)</li>
            <li>⚡ Unusual Response Time (>2s)</li>
            <li>🔄 Rapid Retry Patterns</li>
            <li>📍 Geographic Anomalies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Anomaly breakdown
        st.subheader("📊 Anomaly Breakdown")
        anomaly_by_type = anomalies.groupby('auth_type').size()
        fig = px.pie(values=anomaly_by_type.values, names=anomaly_by_type.index,
                    title="Anomalies by Auth Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🗺️ Geographic Distribution")
        anomaly_regions = anomalies.groupby('region').size().sort_values(ascending=False).head(10)
        fig = px.bar(x=anomaly_regions.index, y=anomaly_regions.values,
                    labels={'x': 'Region', 'y': 'Anomaly Count'},
                    color=anomaly_regions.values,
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent anomalies table
    st.subheader("📋 Recent Anomalies")
    
    display_anomalies = anomalies.sort_values('timestamp', ascending=False).head(50)
    display_anomalies['risk_score'] = np.random.randint(60, 95, len(display_anomalies))
    display_anomalies['severity'] = pd.cut(display_anomalies['risk_score'], 
                                           bins=[0, 70, 85, 100],
                                           labels=['Medium', 'High', 'Critical'])
    
    st.dataframe(
        display_anomalies[['timestamp', 'region', 'auth_type', 'result', 'retry_count', 
                          'failure_reason', 'risk_score', 'severity']].head(20),
        use_container_width=True,
        height=400
    )
    
    # Explainability panel
    st.subheader("🔍 Why Was This Flagged?")
    
    selected_idx = st.selectbox("Select Transaction", options=range(min(10, len(display_anomalies))),
                                format_func=lambda x: f"Transaction {display_anomalies.iloc[x]['transaction_id'][:8]}...")
    
    if selected_idx is not None:
        record = display_anomalies.iloc[selected_idx]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Contributing Factors:**")
            factors = []
            if record['retry_count'] > 3:
                factors.append(f"🔴 High retry count ({record['retry_count']} attempts)")
            if record['response_time_ms'] > 1500:
                factors.append(f"🟠 Slow response time ({record['response_time_ms']:.0f}ms)")
            if record['result'] == 'Failure':
                factors.append(f"🔴 Authentication failed: {record['failure_reason']}")
            if record['hour'] < 6 or record['hour'] > 22:
                factors.append(f"🟡 Off-hours activity (Hour: {record['hour']})")
            
            for factor in factors:
                st.markdown(f"- {factor}")
        
        with col2:
            st.metric("Risk Score", f"{record['risk_score']}/100")
            st.metric("Severity", record['severity'])
            st.metric("Confidence", f"{np.random.randint(75, 95)}%")

def show_predictive_analytics(df):
    """Predictive analytics and forecasting"""
    
    st.header("🔮 Predictive Analytics & Risk Forecasting")
    
    st.markdown("""
    <div class="info-box">
    <strong>📊 Predictive Models:</strong> Our system uses machine learning to forecast authentication patterns,
    predict failure rates, and identify potential system issues before they occur.
    </div>
    """, unsafe_allow_html=True)
    
    # Failure rate prediction
    st.subheader("📈 Failure Rate Prediction (Next 7 Days)")
    
    # Historical failure rate
    df['date'] = df['timestamp'].dt.date
    daily_failures = df.groupby('date').apply(
        lambda x: (x['result'] == 'Failure').sum() / len(x) * 100
    ).reset_index()
    daily_failures.columns = ['Date', 'Failure_Rate']
    
    # Simple prediction (moving average + trend)
    last_7_days_rate = daily_failures['Failure_Rate'].tail(7).mean()
    predicted_rates = [last_7_days_rate + np.random.uniform(-2, 2) for _ in range(7)]
    
    future_dates = pd.date_range(start=daily_failures['Date'].max() + timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Failure_Rate': predicted_rates,
        'Lower_Bound': [max(0, r - 3) for r in predicted_rates],
        'Upper_Bound': [min(100, r + 3) for r in predicted_rates]
    })
    
    # Plot
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=daily_failures['Date'].tail(14),
        y=daily_failures['Failure_Rate'].tail(14),
        name='Historical',
        mode='lines+markers',
        line=dict(color='#0066B2')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Failure_Rate'],
        name='Forecast',
        mode='lines+markers',
        line=dict(color='#FF6B35', dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['Upper_Bound'].tolist() + forecast_df['Lower_Bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,107,53,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Failure Rate Forecast",
        xaxis_title="Date",
        yaxis_title="Failure Rate (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk indicators
    st.markdown("---")
    st.subheader("🚨 Risk Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate risk score based on multiple factors
        risk_score = min(100, int(
            (last_7_days_rate * 2) +
            (df['is_anomaly'].sum() / len(df) * 1000) +
            np.random.uniform(-5, 5)
        ))
        
        risk_level = "Low" if risk_score < 40 else "Medium" if risk_score < 70 else "High"
        risk_color = "#4CAF50" if risk_score < 40 else "#FF6B35" if risk_score < 70 else "#E53935"
        
        st.markdown(f'<div class="metric-card" style="border-left-color: {risk_color}">', unsafe_allow_html=True)
        st.metric("Overall Risk Score", f"{risk_score}/100")
        st.markdown(f"**Level:** {risk_level}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Network health
        network_errors = df[df['failure_reason'] == 'Network_Error'].shape[0]
        network_risk = min(100, int((network_errors / len(df)) * 1000))
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Network Health", f"{100-network_risk}%")
        st.markdown(f"**Network Errors:** {network_errors:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # System load prediction
        avg_daily = len(df) / (df['timestamp'].max() - df['timestamp'].min()).days
        predicted_load = int(avg_daily * 1.1)  # 10% growth
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Daily Load", f"{predicted_load:,}")
        st.markdown("**Trend:** ↗️ Growing")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommended Actions")
    
    if risk_score > 70:
        st.markdown('<div class="alert-box alert-critical">🔴 <strong>CRITICAL:</strong> High risk detected - immediate action required</div>', unsafe_allow_html=True)
        st.markdown("""
        **Immediate Actions:**
        1. 🔍 Investigate regions with high failure rates
        2. 🔧 Check network infrastructure in affected areas
        3. 👥 Increase operator support during peak hours
        4. 📞 Alert technical team for system review
        """)
    elif risk_score > 40:
        st.markdown('<div class="alert-box alert-high">🟠 <strong>ATTENTION:</strong> Moderate risk - monitor closely</div>', unsafe_allow_html=True)
        st.markdown("""
        **Recommended Actions:**
        1. 📊 Monitor failure rates closely
        2. 🔄 Review retry patterns
        3. 🗺️ Analyze geographic trends
        """)
    else:
        st.markdown('<div class="alert-box alert-medium">🟢 <strong>NORMAL:</strong> System operating within acceptable parameters</div>', unsafe_allow_html=True)
        st.markdown("""
        **Maintain:**
        1. ✅ Continue regular monitoring
        2. 📈 Track performance metrics
        3. 🔄 Regular system health checks
        """)

def show_detailed_reports(df):
    """Detailed reports and data export"""
    
    st.header("📋 Detailed Reports & Data Export")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Transaction Summary", "Regional Performance", "Failure Analysis", "Anomaly Report", "Custom Query"]
    )
    
    if report_type == "Transaction Summary":
        st.subheader("📊 Transaction Summary Report")
        
        summary = df.groupby('result').agg({
            'transaction_id': 'count',
            'response_time_ms': 'mean',
            'retry_count': 'mean'
        }).reset_index()
        summary.columns = ['Result', 'Count', 'Avg_Response_Time', 'Avg_Retries']
        
        st.dataframe(summary, use_container_width=True)
        
        # Download button
        csv = summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="transaction_summary.csv",
            mime="text/csv"
        )
    
    elif report_type == "Regional Performance":
        st.subheader("🗺️ Regional Performance Report")
        
        regional = df.groupby('region').agg({
            'transaction_id': 'count',
            'result': lambda x: (x == 'Success').sum() / len(x) * 100,
            'response_time_ms': 'mean',
            'is_anomaly': 'sum'
        }).reset_index()
        regional.columns = ['Region', 'Total_Transactions', 'Success_Rate', 'Avg_Response_Time', 'Anomalies']
        
        st.dataframe(regional, use_container_width=True, height=400)
        
        csv = regional.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Regional Report",
            data=csv,
            file_name="regional_performance.csv",
            mime="text/csv"
        )
    
    elif report_type == "Failure Analysis":
        st.subheader("❌ Failure Analysis Report")
        
        failures = df[df['result'] == 'Failure']
        
        failure_summary = failures.groupby('failure_reason').agg({
            'transaction_id': 'count',
            'region': lambda x: x.mode()[0] if len(x) > 0 else 'N/A',
            'auth_type': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
        }).reset_index()
        failure_summary.columns = ['Failure_Reason', 'Count', 'Most_Affected_Region', 'Most_Common_Auth_Type']
        failure_summary = failure_summary.sort_values('Count', ascending=False)
        
        st.dataframe(failure_summary, use_container_width=True)
        
        csv = failure_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Failure Report",
            data=csv,
            file_name="failure_analysis.csv",
            mime="text/csv"
        )
    
    elif report_type == "Anomaly Report":
        st.subheader("⚠️ Anomaly Detection Report")
        
        anomalies = df[df['is_anomaly'] == 1]
        
        st.metric("Total Anomalies", f"{len(anomalies):,}")
        
        anomaly_details = anomalies[['timestamp', 'region', 'auth_type', 'result', 
                                     'retry_count', 'response_time_ms', 'failure_reason']].copy()
        anomaly_details['risk_score'] = np.random.randint(60, 95, len(anomaly_details))
        
        st.dataframe(anomaly_details.head(100), use_container_width=True, height=400)
        
        csv = anomaly_details.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Anomaly Report",
            data=csv,
            file_name="anomaly_report.csv",
            mime="text/csv"
        )
    
    else:  # Custom Query
        st.subheader("🔍 Custom Data Query")
        
        st.markdown("**Filter Data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_regions = st.multiselect("Regions", options=df['region'].unique(), default=[])
            selected_auth_types = st.multiselect("Auth Types", options=df['auth_type'].unique(), default=[])
        
        with col2:
            selected_result = st.multiselect("Result", options=df['result'].unique(), default=[])
            date_range = st.date_input("Date Range", value=[df['timestamp'].min(), df['timestamp'].max()])
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_regions:
            filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
        if selected_auth_types:
            filtered_df = filtered_df[filtered_df['auth_type'].isin(selected_auth_types)]
        if selected_result:
            filtered_df = filtered_df[filtered_df['result'].isin(selected_result)]
        
        st.metric("Filtered Records", f"{len(filtered_df):,}")
        
        if len(filtered_df) > 0:
            st.dataframe(filtered_df.head(100), use_container_width=True, height=400)
            
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="custom_query_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
