import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title = "Heart Disease Risk Predictor",
    page_icon  = "🫀",
    layout     = "centered"
)

# ── Load Model ────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/xgb_model.pkl')

model = load_model()

# ── Title ─────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Fill in the patient details below to predict disease risk.")
st.divider()

# ── User Input Form ───────────────────────────────
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age      = st.slider("Age", 20, 80, 45)
    sex      = st.selectbox("Sex", options=[0, 1],
                            format_func=lambda x: "Female" if x == 0 else "Male")
    cp       = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120?",
                            options=[0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    restecg  = st.selectbox("Resting ECG Result (0-2)", [0, 1, 2])
    thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina?",
                            options=[0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope    = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
    ca       = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal     = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

st.divider()

# ── Predict Button ────────────────────────────────
if st.button("🔍 Predict Risk", use_container_width=True):

    # build input dataframe
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    # prediction
    prediction    = model.predict(input_data)[0]
    probability   = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    # ── Result Box ────────────────────────────────
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — Probability: {probability:.1%}")
    else:
        st.success(f"✅ LOW RISK — Probability: {probability:.1%}")

    # ── Probability Bar ───────────────────────────
    st.markdown("**Risk Probability:**")
    st.progress(float(probability))

    st.divider()

    # ── SHAP Waterfall Plot ───────────────────────
    st.subheader("🔍 Why this prediction? (SHAP Explanation)")

    explainer   = shap.Explainer(model, input_data)
    shap_values = explainer(input_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.caption("🔴 Red bars push toward HIGH RISK  |  🔵 Blue bars push toward LOW RISK")