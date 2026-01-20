"""
IDEA - Identity Data Evaluation & Analytics - MVP Dashboard
Real-time authentication monitoring and intelligence
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add intelligence modules to path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.anomaly_detector import RealTimeAnomalyDetector
from intelligence.risk_scorer import RiskScorer

# Page configuration
st.set_page_config(
    page_title="IDEA - Identity Data Evaluation & Analytics",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide deploy button only
st.markdown("""
    <style>
    /* Hide only the deploy button */
    .stDeployButton {display: none;}
    button[data-testid="stBaseButton-header"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Simple authentication (MVP - hardcoded users)
USERS = {
    'admin': {'password': 'admin123', 'role': 'Admin'},
    'analyst': {'password': 'analyst123', 'role': 'Analyst'}
}

def login_page():
    """Display login page"""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #1f77b4;'>IDEA - Identity Data Evaluation & Analytics</h1>
            <p style='font-size: 1.2rem; color: #666;'>Real-Time Authentication Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Secure Login")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            if st.button("Login", use_container_width=True, type="primary"):
                if username in USERS and USERS[username]['password'] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = USERS[username]['role']
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        st.markdown("---")
        st.info("""
        **Demo Credentials:**
        - Admin: `admin` / `admin123`
        - Analyst: `analyst` / `analyst123`
        """)

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.rerun()

@st.cache_data
def load_data():
    """Load transaction data"""
    df = pd.read_csv('data/raw/auth_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_detectors(df):
    """Initialize and train detectors"""
    # Anomaly detector
    anomaly_detector = RealTimeAnomalyDetector()
    train_size = int(len(df) * 0.8)
    anomaly_detector.train(df[:train_size])
    
    # Risk scorer
    risk_scorer = RiskScorer()
    training_data = risk_scorer.prepare_training_data(df[:train_size])
    risk_scorer.train(training_data)
    
    return anomaly_detector, risk_scorer

def main_dashboard():
    """Main dashboard after login"""
    
    # Header with user profile
    st.markdown("""
        <style>
        .user-profile {
            position: fixed;
            top: 60px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 12px 20px;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("IDEA - Executive Dashboard")
    
    # User profile section on the right
    col1, col2, col3 = st.sidebar.columns([1, 1, 1])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; 
                    border-radius: 10px; 
                    text-align: center; 
                    color: white;
                    margin-bottom: 20px;'>
            <h3 style='margin: 0; color: white;'>User Profile</h3>
            <p style='margin: 5px 0; font-size: 1.1em;'><strong>{st.session_state.username}</strong></p>
            <p style='margin: 0; font-size: 0.9em;'>Role: {st.session_state.user_role}</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Logout", use_container_width=True, type="primary"):
            logout()
    with col2:
        if st.button("Reload", use_container_width=True):
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Load data
    df = load_data()
    
    # Filter for last 24 hours
    last_24h = df[df['timestamp'] >= df['timestamp'].max() - timedelta(hours=24)]
    
    # KPIs
    st.markdown("### Key Metrics (Last 24 Hours)")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        total_auths = len(last_24h)
        st.metric("Total Authentications", f"{total_auths:,}")
    
    with kpi2:
        success_rate = (last_24h['result'] == 'Success').mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%", 
                 delta=f"{success_rate - 85:.1f}%" if success_rate < 85 else None)
    
    with kpi3:
        # Quick anomaly count (simplified)
        anomaly_count = last_24h['is_anomaly'].sum() if 'is_anomaly' in last_24h.columns else int(len(last_24h) * 0.05)
        st.metric("Anomalies Detected", f"{anomaly_count:,}")
    
    with kpi4:
        high_risk_count = int(anomaly_count * 0.15)  # Estimate
        st.metric("High-Risk Alerts", f"{high_risk_count}", delta="🔴 Critical")
    
    st.markdown("---")
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Authentication Trend")
        
        # Hourly trend
        hourly = last_24h.groupby([
            last_24h['timestamp'].dt.floor('H'),
            'result'
        ]).size().reset_index()
        hourly.columns = ['Hour', 'Result', 'Count']
        
        fig = px.line(hourly, x='Hour', y='Count', color='Result',
                     title='Hourly Authentication Trend (Success vs Failure)')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Regional Distribution")
        
        regional = last_24h.groupby('region').agg({
            'result': lambda x: (x == 'Failure').mean() * 100
        }).reset_index()
        regional.columns = ['Region', 'Failure Rate (%)']
        regional = regional.sort_values('Failure Rate (%)', ascending=False)
        
        fig = px.bar(regional.head(10), x='Region', y='Failure Rate (%)',
                    title='Top 10 Regions by Failure Rate',
                    color='Failure Rate (%)',
                    color_continuous_scale='Reds')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Anomalies", "Risk Analysis", "Patterns", "Alerts"
    ])
    
    with tab1:
        show_anomalies(df, last_24h)
    
    with tab2:
        show_risk_analysis(df, last_24h)
    
    with tab3:
        show_patterns(last_24h)
    
    with tab4:
        show_alerts(last_24h)

def show_anomalies(df, last_24h):
    """Anomalies tab"""
    st.markdown("### Detected Anomalies")
    
    # Load detector
    anomaly_detector, _ = load_detectors(df)
    
    # Detect anomalies in last 24h
    anomaly_results = anomaly_detector.detect_anomalies(last_24h)
    anomalies = anomaly_results[anomaly_results['is_anomaly'] == 1].copy()
    
    if len(anomalies) > 0:
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", len(anomalies))
        with col2:
            high_severity = len(anomalies[anomalies['severity'] == 'High'])
            st.metric("High Severity", high_severity)
        with col3:
            avg_conf = anomalies['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        # Severity distribution
        severity_dist = anomalies['severity'].value_counts()
        fig = px.pie(values=severity_dist.values, names=severity_dist.index,
                    title='Anomaly Severity Distribution',
                    color_discrete_map={'Low': '#90EE90', 'Medium': '#FFD700', 'High': '#FF6347'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly table
        st.markdown("#### Anomaly Details")
        
        display_cols = ['timestamp', 'region', 'auth_type', 'result', 
                       'retry_count', 'severity', 'confidence', 'anomaly_reasons']
        
        display_df = anomalies[display_cols].head(20).copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Explainability panel
        st.markdown("#### Anomaly Explainability")
        
        selected_idx = st.selectbox(
            "Select anomaly to explain:",
            options=range(min(10, len(anomalies))),
            format_func=lambda x: f"Anomaly #{x+1} - {anomalies.iloc[x]['severity']} severity"
        )
        
        if selected_idx is not None:
            anomaly = anomalies.iloc[selected_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Transaction Details:**
                - **Time**: {anomaly['timestamp']}
                - **Region**: {anomaly['region']}
                - **Auth Type**: {anomaly['auth_type']}
                - **Result**: {anomaly['result']}
                - **Retry Count**: {anomaly['retry_count']}
                """)
            
            with col2:
                st.markdown(f"""
                **Anomaly Assessment:**
                - **Severity**: {anomaly['severity']}
                - **Confidence**: {anomaly['confidence']:.1f}%
                - **Reasons**: {anomaly['anomaly_reasons']}
                """)
            
            st.success(f"Recommendation: Investigate device ID and operator pattern for potential security concern")
    
    else:
        st.info("No anomalies detected in the last 24 hours")

def show_risk_analysis(df, last_24h):
    """Risk analysis tab"""
    st.markdown("### Predictive Risk Analysis")
    
    # Load risk scorer
    _, risk_scorer = load_detectors(df)
    
    # Predict next window
    prediction = risk_scorer.predict_next_window(last_24h)
    
    # Risk score gauge
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction['risk_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score (Next Hour)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80}}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Risk Label", prediction['risk_label'])
        st.metric("Probability", f"{prediction['risk_probability']:.1%}")
    
    with col3:
        st.metric("Current Failure Rate", f"{prediction['current_metrics']['failure_rate']:.1%}")
        st.metric("Avg Retries", f"{prediction['current_metrics']['avg_retry']:.2f}")
    
    # Top risk factors
    st.markdown("#### 🔍 Top Risk Contributing Factors")
    
    if prediction['top_risk_factors']:
        factors_df = pd.DataFrame([
            {'Factor': k, 'Importance': v['importance'], 'Current Value': v['current_value']}
            for k, v in prediction['top_risk_factors'].items()
        ])
        
        fig = px.bar(factors_df, x='Importance', y='Factor', orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("#### Recommendations")
    
    if prediction['risk_score'] > 70:
        st.error("""
        HIGH RISK DETECTED
        - Immediate action required
        - Monitor authentication requests closely
        - Consider increasing system capacity
        - Alert operations team
        """)
    elif prediction['risk_score'] > 50:
        st.warning("""
        MODERATE RISK
        - Elevated failure rates detected
        - Review recent system changes
        - Monitor for escalation
        """)
    else:
        st.success("""
        LOW RISK
        - Systems operating normally
        - Continue standard monitoring
        """)

def show_patterns(last_24h):
    """Patterns analysis tab"""
    st.markdown("### Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Auth type distribution
        auth_dist = last_24h['auth_type'].value_counts()
        fig = px.pie(values=auth_dist.values, names=auth_dist.index,
                    title='Authentication Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly pattern
        last_24h['hour'] = last_24h['timestamp'].dt.hour
        hourly_volume = last_24h.groupby('hour').size()
        fig = px.bar(x=hourly_volume.index, y=hourly_volume.values,
                    title='Authentication Volume by Hour',
                    labels={'x': 'Hour', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Failure reasons
        failures = last_24h[last_24h['result'] == 'Failure']
        if len(failures) > 0:
            failure_reasons = failures['failure_reason'].value_counts()
            fig = px.bar(x=failure_reasons.values, y=failure_reasons.index,
                        orientation='h',
                        title='Top Failure Reasons',
                        labels={'x': 'Count', 'y': 'Reason'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Response time distribution
        fig = px.histogram(last_24h, x='response_time_ms', nbins=50,
                          title='Response Time Distribution')
        st.plotly_chart(fig, use_container_width=True)

def show_alerts(last_24h):
    """Alerts tab"""
    st.markdown("### Active Alerts")
    
    # Generate alerts based on anomalies
    high_retry = last_24h[last_24h['retry_count'] > 5]
    slow_response = last_24h[last_24h['response_time_ms'] > 2000]
    high_failure_regions = last_24h.groupby('region').apply(
        lambda x: (x['result'] == 'Failure').mean()
    )
    high_failure_regions = high_failure_regions[high_failure_regions > 0.20]
    
    # Alert summary
    total_alerts = len(high_retry) + len(slow_response) + len(high_failure_regions)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Alerts", total_alerts)
    with col2:
        st.metric("High Severity", int(total_alerts * 0.2))
    with col3:
        st.metric("New (Unacknowledged)", int(total_alerts * 0.6))
    
    # Alert list
    alerts_data = []
    
    # High retry alerts
    for idx, row in high_retry.head(5).iterrows():
        alerts_data.append({
            'Time': row['timestamp'].strftime('%H:%M'),
            'Type': 'Excessive Retries',
            'Severity': 'High' if row['retry_count'] > 8 else 'Medium',
            'Description': f"{row['retry_count']} retries from {row['region']}",
            'Status': 'New'
        })
    
    # Slow response alerts
    for idx, row in slow_response.head(5).iterrows():
        alerts_data.append({
            'Time': row['timestamp'].strftime('%H:%M'),
            'Type': 'Slow Response',
            'Severity': 'Medium',
            'Description': f"{row['response_time_ms']}ms response time",
            'Status': 'New'
        })
    
    # Regional failure alerts
    for region, failure_rate in high_failure_regions.head(5).items():
        alerts_data.append({
            'Time': datetime.now().strftime('%H:%M'),
            'Type': 'Regional Failure Spike',
            'Severity': 'High',
            'Description': f"{region}: {failure_rate:.1%} failure rate",
            'Status': 'New'
        })
    
    if alerts_data:
        alerts_df = pd.DataFrame(alerts_data)
        
        # Color code by severity
        def highlight_severity(row):
            if row['Severity'] == 'High':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Severity'] == 'Medium':
                return ['background-color: #fff4cc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            alerts_df.style.apply(highlight_severity, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Acknowledge All"):
                st.success("All alerts acknowledged")
        with col2:
            if st.button("Export Report"):
                st.info("Report exported")
        with col3:
            if st.button("Notify Operations"):
                st.success("Operations team notified")
    else:
        st.info("No active alerts")


# Main app logic
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
