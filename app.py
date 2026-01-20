import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="UPI Fraud Detection", page_icon="💳", layout="centered")

# -----------------------------
# Load artifact
# -----------------------------
with open("easyensemble_fraud_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
threshold = artifact["threshold"]
encoders = artifact["encoders"]
scaler = artifact["scaler"]
features = artifact["metadata"]["features_expected"]

# -----------------------------
# Session state
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "input"

# -----------------------------
# Global CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #ffffff);
}

.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #555;
}

.result-card {
    padding: 40px;
    border-radius: 22px;
    color: white;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.fraud {
    background: linear-gradient(135deg, #ff4b4b, #b71c1c);
}

.safe {
    background: linear-gradient(135deg, #2ecc71, #1b5e20);
}

.badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 16px;
    margin-top: 12px;
}
.high { background-color: rgba(255,255,255,0.25); }
.low { background-color: rgba(255,255,255,0.25); }
</style>
""", unsafe_allow_html=True)

# =====================================================
# INPUT PAGE
# =====================================================
if st.session_state.page == "input":

    st.markdown('<div class="title">💳 UPI Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter transaction details to assess risk</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    amount = st.number_input("💰 Transaction Amount", min_value=1.0)

    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox("🔄 Transaction Type", ["send", "receive", "merchant_payment"])
        device_type = st.selectbox("📱 Device Type", ["mobile", "tablet"])
        network_type = st.selectbox("🌐 Network Type", ["WiFi", "4G", "5G"])
        month = st.selectbox(
            "📆 Month",
            ["January","February","March","April","May","June",
             "July","August","September","October","November","December"]
        )

    with col2:
        location = st.selectbox(
            "📍 Location",
            ["Mumbai","Delhi","Bangalore","Hyderabad",
             "Chennai","Kolkata","Pune","Ahmedabad"]
        )
        time_of_day = st.selectbox("⏰ Time of Day", ["morning","afternoon","evening","night"])
        day_of_week = st.selectbox(
            "📅 Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday",
             "Friday","Saturday","Sunday"]
        )

    is_rooted_device = st.radio("🔓 Rooted Device", ["No", "Yes"], horizontal=True)
    day = st.slider("📅 Day of Month", 1, 31)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Analyze Transaction", use_container_width=True):

        df = pd.DataFrame([{
            "amount": amount,
            "transaction_type": transaction_type,
            "location": location,
            "device_type": device_type,
            "network_type": network_type,
            "time_of_day": time_of_day,
            "month": month,
            "day_of_week": day_of_week,
            "is_rooted_device": 1 if is_rooted_device == "Yes" else 0,
            "day": day
        }])

        for col, le in encoders.items():
            df[col] = le.transform(df[col])

        df[["amount","day"]] = scaler.transform(df[["amount","day"]])
        df = df[features]

        prob = model.predict_proba(df)[:,1][0]

        st.session_state.prob = prob
        st.session_state.is_fraud = prob >= threshold
        st.session_state.page = "result"
        st.rerun()

# =====================================================
# RESULT PAGE
# =====================================================
else:
    prob = st.session_state.prob
    is_fraud = st.session_state.is_fraud

    if is_fraud:
        st.markdown(f"""
        <div class="result-card fraud">
            🚨 FRAUD DETECTED<br><br>
            Probability: {prob:.2f}
            <div class="badge high">HIGH RISK</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card safe">
            ✅ TRANSACTION SAFE<br><br>
            Probability: {prob:.2f}
            <div class="badge low">LOW RISK</div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(min(prob,1.0))
    st.caption(f"Decision Threshold: {threshold}")

    if st.button("🔁 Check Another Transaction", use_container_width=True):
        st.session_state.page = "input"
        st.rerun()
