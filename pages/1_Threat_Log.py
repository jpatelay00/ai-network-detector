import streamlit as st
import sqlite3
import pandas as pd
import json
import time

st.set_page_config(page_title="Historical Threat Log", layout="wide")
st.title("🗄️ Historical Threat Database")
st.markdown("Review, filter, and manage anomalous connections flagged by the AI Network Anomaly Detector.")

def load_alerts():
    conn = sqlite3.connect('threat_alerts.db')
    # Read the database into a Pandas DataFrame, ordered by most recent first
    df = pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC", conn)
    conn.close()
    return df

alerts_df = load_alerts()

if alerts_df.empty:
    st.success("No historical threats found. The network is secure.")
else:
    st.subheader("Logged Security Alerts")
    
    # 1. Provide filtering options
    severity_filter = st.selectbox("Filter by Severity", ["All", "Critical", "High", "Medium", "Low"])
    
    if severity_filter != "All":
        display_df = alerts_df[alerts_df['severity'] == severity_filter]
    else:
        display_df = alerts_df
        
    st.dataframe(display_df[['id', 'timestamp', 'severity', 'risk_score']], use_container_width=True)
    
    st.divider()
    st.subheader("Deep Packet Inspection")
    
    if not display_df.empty:
        # 2. Allow the user to select a specific alert to inspect
        selected_id = st.selectbox("Select Alert ID to inspect details:", display_df['id'])
        
        if selected_id:
            # Extract the JSON string from the database and convert it back to a Python dictionary
            details_str = display_df[display_df['id'] == selected_id]['details'].values[0]
            details_dict = json.loads(details_str)
            
            # Display the raw JSON packet data
            st.json(details_dict)
            
            # 3. Provide an action to clear the alert (Delete operation)
            if st.button("🚨 Mark as Investigated (Delete Alert)"):
                conn = sqlite3.connect('threat_alerts.db')
                c = conn.cursor()
                c.execute("DELETE FROM alerts WHERE id=?", (selected_id,))
                conn.commit()
                conn.close()
                st.success(f"Alert {selected_id} marked as investigated and removed from the database.")
                time.sleep(1) # Give the user a second to read the success message
                st.rerun() # Refresh the page to update the table
    else:
        st.info("No alerts match the current filter.")