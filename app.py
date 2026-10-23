import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="AI Network Anomaly Detector", layout="wide")

st.title("🛡️ AI Network Anomaly Detector")
st.markdown("""
This application uses a trained **Isolation Forest** machine learning model to analyze network traffic and identify anomalous connections.
*Note: Anomalous traffic indicates unusual behavior, which may warrant further investigation by security teams, but does not always guarantee a cyberattack.*
""")

# 2. Load the AI Model and Scaler efficiently
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

# 3. File Upload UI
st.divider()
st.subheader("1. Upload Network Traffic")
uploaded_file = st.file_uploader("Upload a CSV file containing network data:", type=['csv'])

if uploaded_file is not None:
    st.info("File uploaded successfully. Analyzing data...")
    df_raw = pd.read_csv(uploaded_file)
    
    missing_cols = [col for col in expected_cols if col not in df_raw.columns]
    
    if len(missing_cols) > 0:
        st.error(f"❌ Uploaded CSV is missing {len(missing_cols)} required columns.")
    else:
        # Preprocess and Predict
        X_new = df_raw[expected_cols]
        X_scaled = scaler.transform(X_new)
        
        predictions = model.predict(X_scaled) 
        scores = model.decision_function(X_scaled) 
        
        df_results = df_raw.copy()
        df_results['AI_Prediction'] = ["Anomaly" if p == -1 else "Normal" for p in predictions]
        df_results['Anomaly_Score'] = scores
        
        total_connections = len(df_results)
        anomalies = len(df_results[df_results['AI_Prediction'] == 'Anomaly'])
        anomaly_pct = (anomalies / total_connections) * 100 if total_connections > 0 else 0
        
        # Dashboard UI
        st.subheader("2. Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Connections Analyzed", total_connections)
        col2.metric("🚨 Anomalies Flagged", anomalies)
        col3.metric("Anomaly Percentage", f"{anomaly_pct:.2f}%")
        
        col_chart, col_table = st.columns([1, 2])
        
        with col_chart:
            st.markdown("**Traffic Classification Breakdown**")
            st.bar_chart(df_results['AI_Prediction'].value_counts(), color="#ff4b4b")
            
        with col_table:
            st.markdown("**Flagged Anomalous Connections (Preview)**")
            df_anomalies = df_results[df_results['AI_Prediction'] == 'Anomaly']
            st.dataframe(df_anomalies.head(50), use_container_width=True)
            
        # --- NEW PHASE 9 CODE: SHAP EXPLAINABILITY ---
        st.divider()
        st.subheader("3. Explainable AI (Why was this flagged?)")
        
        if anomalies > 0:
            st.markdown("Select an anomalous connection from the dropdown to see exactly which network features triggered the AI's alarm.")
            
            # Let the user pick a specific anomaly by its row index
            anomaly_indices = df_anomalies.index.tolist()
            selected_idx = st.selectbox("Select Anomaly Row Index to Analyze:", anomaly_indices)
            
            if selected_idx is not None:
                # Find exactly where this row lives in our scaled math data
                row_position = df_raw.index.get_loc(selected_idx)
                single_observation_scaled = X_scaled[row_position].reshape(1, -1)
                
                with st.spinner("Generating Game Theory explanations (SHAP)..."):
                    # 1. Ask SHAP to explain the model's decision
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(single_observation_scaled)
                    
                    # 2. Convert to a Pandas Series so we can easily plot it
                    shap_series = pd.Series(shap_values[0], index=expected_cols)
                    
                    # 3. Sort by absolute magnitude to find the most impactful features
                    shap_series = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)
                    top_shap = shap_series.head(10) # Just grab the top 10 culprits
                    
                    # 4. Build a beautiful Matplotlib chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Red means it pushed the score lower (towards anomaly). Green means towards normal.
                    colors = ['#ff4b4b' if val < 0 else '#00cc66' for val in top_shap.values]
                    
                    top_shap.plot(kind='barh', color=colors, ax=ax)
                    ax.invert_yaxis() # Put the biggest impact at the top
                    ax.set_title(f"Top 10 Features Driving the Anomaly Alert (Row {selected_idx})")
                    ax.set_xlabel("SHAP Value (Impact on Anomaly Score)")
                    
                    st.pyplot(fig)
                    
                    st.markdown("""
                    **How to read this chart:**
                    * The **Isolation Forest** assigns negative scores to anomalies. 
                    * 🟥 **Red bars (Negative SHAP):** These network features actively pushed the AI to flag this connection as a cyberattack.
                    * 🟩 **Green bars (Positive SHAP):** These features actually looked normal, but were overpowered by the red features.
                    """)
        else:
            st.success("No anomalies detected in this batch! Nothing to explain.")

        # Export Results
        st.divider()
        st.subheader("4. Export Results")
        csv_export = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Analysis (CSV)",
            data=csv_export,
            file_name='network_analysis_results.csv',
            mime='text/csv',
        )