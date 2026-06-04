import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Load Artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = tf.keras.models.load_model('churn_model.keras')
    scaler    = pickle.load(open('scaler.pkl', 'rb'))
    threshold = pickle.load(open('threshold.pkl', 'rb'))
    cols      = pickle.load(open('feature_cols.pkl', 'rb'))
    return model, scaler, threshold, cols

model, scaler, threshold, feature_cols = load_artifacts()

# ── UI ─────────────────────────────────────────────────
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict churn risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    credit_score     = st.slider("Credit Score",       300, 850, 650)
    age              = st.slider("Age",                18,  92,  35)
    tenure           = st.slider("Tenure (years)",     0,   10,  5)
    balance          = st.number_input("Balance ($)",  min_value=0.0, value=50000.0, step=1000.0)
    num_products     = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    has_cr_card      = st.selectbox("Has Credit Card",    ["Yes", "No"])
    is_active        = st.selectbox("Is Active Member",   ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)
    geography        = st.selectbox("Geography",          ["France", "Germany", "Spain"])
    gender           = st.selectbox("Gender",             ["Male", "Female"])

st.divider()

# ── Predict Button ─────────────────────────────────────
if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):

    # Build input matching exact training columns
    input_dict = {
        'CreditScore':        credit_score,
        'Age':                age,
        'Tenure':             tenure,
        'Balance':            balance,
        'NumOfProducts':      num_products,
        'HasCrCard':          1 if has_cr_card == "Yes" else 0,
        'IsActiveMember':     1 if is_active   == "Yes" else 0,
        'EstimatedSalary':    estimated_salary,
        'Geography_Germany':  1 if geography == "Germany" else 0,
        'Geography_Spain':    1 if geography == "Spain"   else 0,
        'Gender_Male':        1 if gender    == "Male"    else 0,
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]  # enforce column order
    input_sc = scaler.transform(input_df)
    prob     = float(model.predict(input_sc, verbose=0)[0][0])
    label    = "Will Churn" if prob >= threshold else "Will Stay"

    # ── Result Display ─────────────────────────────────
    st.subheader("Prediction Result")

    if prob >= threshold:
        st.error(f"⚠️ **{label}**  —  Churn Probability: **{prob*100:.1f}%**")
    else:
        st.success(f"✅ **{label}**  —  Churn Probability: **{prob*100:.1f}%**")

    # Progress bar as risk gauge
    st.markdown("**Risk Gauge**")
    st.progress(prob)

    # Breakdown
    st.markdown("**Key Inputs Summary**")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Age",         age)
    summary_cols[1].metric("Balance",     f"${balance:,.0f}")
    summary_cols[2].metric("Geography",   geography)
    summary_cols[3].metric("Active?",     is_active)

    st.caption(f"Decision threshold: {threshold:.3f} | Model AUC: 0.861")