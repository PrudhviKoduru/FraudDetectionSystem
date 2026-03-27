import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

# Configure the visual layout
st.set_page_config(page_title="Fraud Mitigation Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Financial Fraud Mitigation System")
# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    /* Style the main background and text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Make the Tabs look like modern pill-buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB !important; /* A nice professional blue */
        color: white !important;
        border: 1px solid #2E86AB;
    }
    
    /* Upgrade the Primary Predict Button */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF3333;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    
    /* Style the input boxes */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #1E222B;
        color: white;
        border-radius: 5px;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA AND MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("scaler.pkl")
        iso = joblib.load("isolation_forest.pkl")
        xgb_model = joblib.load("xgboost_model.pkl")
        return scaler, iso, xgb_model
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("transactions_mapped.csv")
        df["Transaction Time"] = pd.to_datetime(df["Transaction Time"])
        df = df.sort_values("Transaction Time").reset_index(drop=True)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

scaler, iso, model = load_models()
history_df = load_historical_data()

# --- UI NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🔴 Live Detection", "📂 Batch Processing", "⚙️ Data & Model Management"])

# ==========================================
# TAB 1: LIVE SINGLE TRANSACTION DETECTION
# ==========================================
with tab1:
    st.subheader("Enter Live Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        device_ip = st.text_input("Device IP", value="49.207.45.128")
        transaction_id = st.text_input("Transaction ID", value="TXN_NEW_123")
        location = st.text_input("Location Name", value="MG Road")

    with col2:
        txn_time_str = st.text_input("Transaction Time (YYYY-MM-DD HH:MM:SS)", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        txn_type = st.selectbox("Transaction Type", ["Debit", "Credit"])
        amount_spike = st.selectbox("Amount Spike Detected?", ["No", "Yes"])

    with col3:
        debited_amt = st.number_input("Debited Amount (INR)", min_value=0.0, value=1500.0)
        credited_amt = st.number_input("Credited Amount (INR)", min_value=0.0, value=0.0)

    if st.button("Predict Transaction Safety", type="primary"):
        if history_df.empty or model is None:
            st.error("System offline: Models or Historical Data missing. Please go to the Data Management tab.")
        else:
            new_time = pd.to_datetime(txn_time_str)
            last_txn = history_df.iloc[-1]
            last_time = last_txn['Transaction Time']

            # Dynamic Feature Engineering
            hour = new_time.hour
            time_gap_sec = max(0, (new_time - last_time).total_seconds())
            
            one_hour_ago = new_time - pd.Timedelta(hours=1)
            velocity_1h = history_df[(history_df['Transaction Time'] >= one_hour_ago) & 
                                     (history_df['Transaction Time'] <= new_time)].shape[0] + 1
            
            five_min_ago = new_time - pd.Timedelta(minutes=5)
            velocity_5min = history_df[(history_df['Transaction Time'] >= five_min_ago) & 
                                       (history_df['Transaction Time'] <= new_time)].shape[0] + 1
            
            ip_change_flag = 1 if device_ip != last_txn['Device IP'] else 0
            location_change_flag = 1 if location != last_txn['Location'] else 0
            amount_spike_flag = 1 if amount_spike == "Yes" else 0
            
            current_total_amt = last_txn['Total Amt'] + credited_amt - debited_amt
            balance_drain_pct = debited_amt / (current_total_amt + 1)
            
            last_9_debits = history_df['Debited Amt'].tail(9).tolist()
            last_9_debits.append(debited_amt)
            rolling_amt_mean = np.mean(last_9_debits)
            
            amt_deviation = debited_amt - rolling_amt_mean
            amt_dev_ratio = debited_amt / (rolling_amt_mean + 1)
            
            ip_history = history_df[history_df['Device IP'] == device_ip]
            ip_time_gap = max(0, (new_time - ip_history.iloc[-1]['Transaction Time']).total_seconds()) if not ip_history.empty else 0.0

            base_features = pd.DataFrame({
                "Debited Amt": [debited_amt], "hour": [hour], "time_gap_sec": [time_gap_sec],
                "velocity_1h": [velocity_1h], "velocity_5min": [velocity_5min], "ip_time_gap": [ip_time_gap],
                "balance_drain_pct": [balance_drain_pct], "ip_change_flag": [ip_change_flag],
                "location_change_flag": [location_change_flag], "Amount_Spike_Flag": [amount_spike_flag],
                "amt_deviation": [amt_deviation], "amt_dev_ratio": [amt_dev_ratio]
            })

            try:
                scaled_features = scaler.transform(base_features)
                iso_score = -iso.decision_function(scaled_features)
                
                final_features = base_features.copy()
                final_features["iso_score"] = iso_score
                
                prediction = model.predict(final_features)[0]
                probability = model.predict_proba(final_features)[0][1]
                
                if prediction == 1:
                     st.error(f"🚨 **FRAUD DETECTED: Transaction blocked.** (Confidence: {probability:.2%})")
                else:
                     st.success(f"✅ **SAFE: Transaction approved.** (Fraud Risk: {probability:.2%})")
                     
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ==========================================
# TAB 2: BATCH PROCESSING
# ==========================================
with tab2:
    st.subheader("Batch Transaction Upload")
    st.info("Upload a CSV file containing pre-engineered transaction features to scan multiple records at once.")
    
    uploaded_batch = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_upload")
    
    if uploaded_batch is not None and model is not None:
        batch_df = pd.read_csv(uploaded_batch)
        st.write("### Data Preview", batch_df.head())
        
        if st.button("Run Batch Prediction"):
            try:
                # Assuming the uploaded batch has the 13 required features already
                predictions = model.predict(batch_df)
                batch_df['Prediction'] = ["Blocked (Fraud)" if p == 1 else "Safe" for p in predictions]
                st.write("### Prediction Results")
                st.dataframe(batch_df)
                
                csv_export = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", data=csv_export, file_name='batch_predictions.csv', mime='text/csv')
            except Exception as e:
                st.error(f"Batch prediction failed. Ensure columns match the model requirements. Error: {e}")

# ==========================================
# TAB 3: DATA & MODEL MANAGEMENT
# ==========================================
with tab3:
    st.subheader("System Overview & Data")
    
    if not history_df.empty:
        # Calculate some quick stats for the dashboard
        total_txns = len(history_df)
        
        # We need to check if 'is_fraud' or 'Purpose' exists to count frauds
        if "is_fraud" in history_df.columns:
            fraud_txns = history_df["is_fraud"].sum()
        elif "Purpose" in history_df.columns:
            fraud_txns = history_df["Purpose"].str.contains("fraud", case=False).astype(int).sum()
        else:
            fraud_txns = 0
            
        safe_txns = total_txns - fraud_txns
        fraud_rate = (fraud_txns / total_txns) * 100 if total_txns > 0 else 0

        # Build the Heads-Up Display (HUD)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total_txns:,}")
        col2.metric("Safe Transactions", f"{safe_txns:,}")
        col3.metric("Blocked (Fraud)", f"{fraud_txns:,}", delta="High Risk", delta_color="inverse")
        col4.metric("System Fraud Rate", f"{fraud_rate:.2f}%")
        
        st.markdown("---")
        st.write("### Recent Transaction Logs")
        st.dataframe(history_df.tail(10), use_container_width=True)
    else:
        st.warning("No historical data found.")

    st.markdown("---")
    
    # Put the upload and retrain sections side-by-side to save vertical space
    manage_col1, manage_col2 = st.columns(2)
    
    with manage_col1:
        st.write("### Add New Transactions")
        new_data_upload = st.file_uploader("Upload new raw transactions (CSV)", type=["csv"], key="new_data")
        if new_data_upload is not None:
            new_df = pd.read_csv(new_data_upload)
            if st.button("Append Data to Database"):
                new_df.to_csv("transactions_mapped.csv", mode='a', header=not os.path.exists("transactions_mapped.csv"), index=False)
                st.cache_data.clear()
                st.success("Data appended! Refresh the page.")

    with manage_col2:
        st.write("### System Retraining Pipeline")
        st.info("Retrain the ML models on the latest dataset.")
        st.write("") # Spacer
        if st.button("Initiate Model Retraining 🚀", use_container_width=True):
            # ... (Keep your existing retraining logic here) ...
            st.success("✅ Models retrained!")
