import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration (must be the first Streamlit command)
st.set_page_config(page_title="AI Network Anomaly Detector", layout="wide")

st.title("🛡️ AI Network Anomaly Detector")
st.markdown("""
This application uses a trained **Isolation Forest** machine learning model to analyze network traffic and identify anomalous connections.
*Note: Anomalous traffic indicates unusual behavior, which may warrant further investigation by security teams, but does not always guarantee a cyberattack.*
""")

# 2. Load the AI Model and Scaler efficiently
# @st.cache_resource ensures the model only loads once, keeping the dashboard fast!
@st.cache_resource
def load_ai_assets():
    model = joblib.load('models/isolation_forest.joblib')
    scaler = joblib.load('models/scaler.joblib')
    # We peek at the training data to know exactly which columns the AI expects
    expected_cols = pd.read_csv('data/X_train.csv', nrows=0).columns.tolist()
    return model, scaler, expected_cols

try:
    model, scaler, expected_cols = load_ai_assets()
except Exception as e:
    st.error("⚠️ Error loading model assets. Please ensure Phases 3 and 4 were completed successfully.")
    st.stop()

# 3. File Upload UI
st.divider()
st.subheader("1. Upload Network Traffic")
uploaded_file = st.file_uploader("Upload a CSV file containing network data:", type=['csv'])

if uploaded_file is not None:
    st.info("File uploaded successfully. Analyzing data...")
    df_raw = pd.read_csv(uploaded_file)
    
    # 4. Feature Validation
    # We must ensure the uploaded CSV has the columns our AI was trained on
    missing_cols = [col for col in expected_cols if col not in df_raw.columns]
    
    if len(missing_cols) > 0:
        st.error(f"❌ Uploaded CSV is missing {len(missing_cols)} required columns.")
        st.write("Missing columns include:", missing_cols[:5], "...and more.")
    else:
        # 5. Preprocessing the new data
        # We extract ONLY the numerical columns the AI expects, in the exact right order
        X_new = df_raw[expected_cols]
        
        # We scale the new data using the mathematical rules saved from Phase 3
        X_scaled = scaler.transform(X_new)
        
        # 6. AI Predictions
        predictions = model.predict(X_scaled) # Outputs 1 (normal) or -1 (anomaly)
        scores = model.decision_function(X_scaled) # Outputs a raw anomaly score
        
        # 7. Formatting the Results
        df_results = df_raw.copy()
        df_results['AI_Prediction'] = ["Anomaly" if p == -1 else "Normal" for p in predictions]
        df_results['Anomaly_Score'] = scores
        
        # Calculate Dashboard Metrics
        total_connections = len(df_results)
        anomalies = len(df_results[df_results['AI_Prediction'] == 'Anomaly'])
        anomaly_pct = (anomalies / total_connections) * 100
        
        # 8. Rendering the Dashboard UI
        st.subheader("2. Analysis Results")
        
        # Display high-level metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Connections Analyzed", total_connections)
        col2.metric("🚨 Anomalies Flagged", anomalies)
        col3.metric("Anomaly Percentage", f"{anomaly_pct:.2f}%")
        
        # Display visualizations and tables side-by-side
        col_chart, col_table = st.columns([1, 2])
        
        with col_chart:
            st.markdown("**Traffic Classification Breakdown**")
            # Create a simple bar chart of the predictions
            st.bar_chart(df_results['AI_Prediction'].value_counts(), color="#ff4b4b")
            
        with col_table:
            st.markdown("**Flagged Anomalous Connections (Preview)**")
            # Filter the dataframe to only show the anomalies
            df_anomalies = df_results[df_results['AI_Prediction'] == 'Anomaly']
            st.dataframe(df_anomalies.head(50), use_container_width=True)
            
        # 9. Download the final report
        st.divider()
        st.subheader("3. Export Results")
        st.markdown("Download the complete analysis, including AI predictions and anomaly scores appended to the original data.")
        
        csv_export = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Analysis (CSV)",
            data=csv_export,
            file_name='network_analysis_results.csv',
            mime='text/csv',
        )