import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import time
import plotly.express as px
import sqlite3
import json
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="AI Network Anomaly Detector", layout="wide")

st.title("🛡️ AI Network Anomaly Detector (Live SOC)")
st.markdown("This application uses a trained **Isolation Forest** machine learning model to analyze network traffic and identify anomalous connections.")

# --- NEW: INITIALIZE SQLITE DATABASE ---
def init_db():
    conn = sqlite3.connect('threat_alerts.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  severity TEXT,
                  risk_score REAL,
                  details TEXT)''')
    conn.commit()
    conn.close()

init_db()

def log_threats_to_db(threat_df):
    conn = sqlite3.connect('threat_alerts.db')
    c = conn.cursor()
    for index, row in threat_df.iterrows():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        severity = row['Severity']
        risk = row['Risk_Score']
        details = row.to_json()
        c.execute("INSERT INTO alerts (timestamp, severity, risk_score, details) VALUES (?, ?, ?, ?)",
                  (timestamp, severity, risk, details))
    conn.commit()
    conn.close()

@st.cache_resource
def load_ai_assets():
    model = joblib.load('models/isolation_forest.joblib')
    scaler = joblib.load('models/scaler.joblib')
    expected_cols = pd.read_csv('data/X_train.csv', nrows=0).columns.tolist()
    return model, scaler, expected_cols

try:
    model, scaler, expected_cols = load_ai_assets()
except Exception as e:
    st.error("⚠️ Error loading model assets. Please ensure Phases 3 and 4 were completed successfully.")
    st.stop()

def calculate_risk_score(raw_score):
    risk = (0.5 - raw_score) * 100
    return max(0, min(100, risk))

st.divider()
st.subheader("1. Connect Network Feed")
uploaded_file = st.file_uploader("Upload a CSV file to simulate a live network feed:", type=['csv'])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    missing_cols = [col for col in expected_cols if col not in df_raw.columns]
    
    if len(missing_cols) > 0:
        st.error(f"❌ Uploaded CSV is missing {len(missing_cols)} required columns.")
    else:
        # Pre-process
        X_new = df_raw[expected_cols]
        X_scaled = pd.DataFrame(scaler.transform(X_new), columns=expected_cols)
        
        predictions = model.predict(X_scaled) 
        scores = model.decision_function(X_scaled) 
        
        df_results = df_raw.copy()
        df_results['AI_Prediction'] = ["Anomaly" if p == -1 else "Normal" for p in predictions]
        df_results['Raw_Score'] = scores
        df_results['Risk_Score'] = [calculate_risk_score(s) for s in scores]
        df_results['Severity'] = pd.cut(
            df_results['Risk_Score'], 
            bins=[-1, 25, 50, 75, 101], 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        st.divider()
        st.subheader("2. Live Operations Monitor")
        
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        table_placeholder = st.empty()
        
        if st.button("🔴 Start Live SOC Stream"):
            demo_limit = min(100, len(df_results))
            
            for i in range(1, demo_limit + 1):
                current_feed = df_results.iloc[:i]
                
                total_conn = len(current_feed)
                anoms = len(current_feed[current_feed['AI_Prediction'] == 'Anomaly'])
                pct = (anoms / total_conn) * 100 if total_conn > 0 else 0
                
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Packets Analyzed", total_conn)
                    if anoms > 0:
                        col2.metric("🚨 Active Threats", anoms, delta="Threat Detected", delta_color="inverse")
                    else:
                        col2.metric("🟢 Active Threats", anoms)
                    col3.metric("Threat Level", f"{pct:.1f}%")
                
                with chart_placeholder.container():
                    fig = px.pie(
                        current_feed, 
                        names='AI_Prediction', 
                        hole=0.4,
                        color='AI_Prediction',
                        color_discrete_map={'Normal': '#00cc66', 'Anomaly': '#ff4b4b'},
                        title="Traffic Classification Distribution"
                    )
                    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with table_placeholder.container():
                    display_cols = ['AI_Prediction', 'Severity', 'Risk_Score'] + expected_cols[:5]
                    st.dataframe(current_feed[display_cols].tail(10), use_container_width=True)
                
                time.sleep(0.3)
            
            # --- NEW: SAVE THREATS TO SQLITE DATABASE ---
            anomalous_data = df_results.iloc[:demo_limit][df_results.iloc[:demo_limit]['AI_Prediction'] == 'Anomaly']
            if not anomalous_data.empty:
                log_threats_to_db(anomalous_data)
                
            st.success("Live stream simulation complete. Threats successfully logged to the database.")
        
        st.divider()
        st.subheader("3. Explainable AI (Threat Attribution)")
        anomalies_total = len(df_results[df_results['AI_Prediction'] == 'Anomaly'])
        
        if anomalies_total > 0:
            df_anomalies = df_results[df_results['AI_Prediction'] == 'Anomaly']
            anomaly_indices = df_anomalies.index.tolist()
            selected_idx = st.selectbox("Select Anomaly Row Index to Analyze:", anomaly_indices)
            
            if selected_idx is not None:
                threat_data = df_anomalies.loc[selected_idx]
                st.warning(f"**Selected Alert Severity:** {threat_data['Severity']} (Risk Score: {threat_data['Risk_Score']:.1f}/100)")
                
                row_position = df_raw.index.get_loc(selected_idx)
                single_observation_scaled = X_scaled.iloc[row_position].values.reshape(1, -1)
                
                with st.spinner("Generating Game Theory explanations (SHAP)..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(single_observation_scaled)
                    
                    shap_series = pd.Series(shap_values[0], index=expected_cols)
                    shap_series = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)
                    top_shap = shap_series.head(10)
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.title.set_color('white')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')

                    colors = ['#ff4b4b' if val < 0 else '#00cc66' for val in top_shap.values]
                    top_shap.plot(kind='barh', color=colors, ax=ax)
                    ax.invert_yaxis() 
                    ax.set_title(f"Top 10 Features Driving the Anomaly Alert")
                    ax.set_xlabel("SHAP Value (Impact on Anomaly Score)")
                    st.pyplot(fig)

                    # --- NEW PHASE 13 CODE: HUMAN-READABLE EXPLANATION ---
                    st.markdown("### 📝 AI Analyst Summary")
                    
                    # Filter for features that actively pushed the score toward Anomaly (negative SHAP values)
                    anomaly_drivers = top_shap[top_shap < 0]
                    
                    if not anomaly_drivers.empty:
                        # Grab the names of the top 3 biggest offenders
                        top_3_offenders = anomaly_drivers.head(3).index.tolist()
                        
                        # Format them nicely into a string
                        offenders_str = ", ".join([f"**{feat}**" for feat in top_3_offenders])
                        
                        st.error(f"🧠 **AI Conclusion:** This connection was flagged with a **{threat_data['Severity']}** severity alert primarily because its network behavior regarding {offenders_str} deviated significantly from established baseline patterns.")
                    else:
                        st.info("The AI detected an anomaly, but the exact driving features are highly distributed across the packet.")
        else:
            st.success("No anomalies detected in this batch! Nothing to explain.")