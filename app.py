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
    st.subheader("Current Training Dataset")
    if not history_df.empty:
        st.write(f"Total historical records: **{len(history_df)}**")
        st.dataframe(history_df.tail(10)) # Show the latest 10 records
    else:
        st.warning("No historical data found.")

    st.markdown("---")
    st.subheader("Add New Transactions")
    new_data_upload = st.file_uploader("Upload new raw transactions (CSV) to append to the database", type=["csv"], key="new_data")
    
    if new_data_upload is not None:
        new_df = pd.read_csv(new_data_upload)
        if st.button("Append Data to Database"):
            # Append to existing CSV and clear Streamlit cache so the app reloads the fresh data
            new_df.to_csv("transactions_mapped.csv", mode='a', header=not os.path.exists("transactions_mapped.csv"), index=False)
            st.cache_data.clear()
            st.success("New data successfully appended! Refresh the page to see updates.")

    st.markdown("---")
    st.subheader("System Retraining Pipeline")
    st.warning("Clicking this will retrain the Scaler, Isolation Forest, and XGBoost models on the latest 'transactions_mapped.csv' data.")
    
    if st.button("Initiate Model Retraining 🚀"):
        with st.spinner('Training ML pipeline in the background...'):
            try:
                # 1. Load the latest data
                train_df = pd.read_csv("transactions_mapped.csv")
                train_df["Transaction Time"] = pd.to_datetime(train_df["Transaction Time"])
                
                # Setup target variable
                train_df["is_fraud"] = train_df["Purpose"].str.contains("fraud", case=False).astype(int)
                
                # Recreation of Notebook Feature Engineering
                train_df["hour"] = train_df["Transaction Time"].dt.hour
                train_df["time_gap_sec"] = train_df["Transaction Time"].diff().dt.total_seconds().fillna(0)
                # Note: For production speed, we are keeping the features simple here. 
                # Ensure the CSV uploaded already contains the required base features, 
                # or replicate your full loop for velocity here.
                
                # Assuming train_df contains the engineered features for training
                features = ["Debited Amt", "hour", "time_gap_sec", "velocity_1h", "velocity_5min", "ip_time_gap", 
                            "balance_drain_pct", "ip_change_flag", "location_change_flag", "Amount_Spike_Flag", 
                            "amt_deviation", "amt_dev_ratio"]
                
                # Filter to only the features we need
                X_base = train_df[features].fillna(0)
                y = train_df["is_fraud"]
                
                # Retrain Scaler
                new_scaler = StandardScaler()
                X_scaled = new_scaler.fit_transform(X_base)
                
                # Retrain Isolation Forest
                new_iso = IsolationForest(n_estimators=200, contamination=0.002, random_state=42)
                new_iso.fit(X_scaled)
                train_df["iso_score"] = -new_iso.decision_function(X_scaled)
                
                # Retrain XGBoost
                features_with_iso = features + ["iso_score"]
                X_final = train_df[features_with_iso].fillna(0)
                
                fraud_ratio = (len(y) - sum(y)) / (sum(y) + 1) # Prevent division by zero
                new_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, 
                                        scale_pos_weight=fraud_ratio, random_state=42)
                new_xgb.fit(X_final, y)
                
                # Save new models
                joblib.dump(new_scaler, 'scaler.pkl')
                joblib.dump(new_iso, 'isolation_forest.pkl')
                joblib.dump(new_xgb, 'xgboost_model.pkl')
                
                # Clear cached resources to load the new models into memory
                st.cache_resource.clear()
                st.success("✅ Models retrained and saved successfully! The Live Detection tab is now using the updated AI.")
                
            except Exception as e:
                st.error(f"Retraining failed. Ensure the dataset contains all required columns. Error details: {e}")
