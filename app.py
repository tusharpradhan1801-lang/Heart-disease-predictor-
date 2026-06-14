import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. SET UP PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Prediction App")
st.write("Enter the patient's clinical details below to calculate risk status.")

# --- 2. LOAD TRAINED ASSETS ---
@st.cache_resource
def load_model_and_scaler():
    # Load your tuned XGBoost model
    model = joblib.load('models/xgb_model.pkl')
    # Tip: Since you scaled features in cell [19], it's best to save that fitted scaler 
    # in your notebook using joblib.dump(scaler, 'models/scaler.pkl') and load it here.
    # For now, we will simulate or use the loaded model.
    try:
        scaler = joblib.load('models/scaler.pkl')
    except:
        scaler = None
    return model, scaler

model, scaler = load_model_and_scaler()

# --- 3. CREATE INPUT USER INTERFACE ---
st.header("Patient Clinical Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", 
                      options=["asymptomatic", "atypical angina", "non-anginal", "typical angina"])
    trestbps = st.number_input("Resting Blood Pressure (trestbps in mm Hg)", min_value=50, max_value=250, value=130)

with col2:
    chol = st.number_input("Serum Cholesterol (chol in mg/dl)", min_value=50, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=["False", "True"])
    restecg = st.selectbox("Resting ECG Results (restecg)", 
                           options=["normal", "lv hypertrophy", "st-t abnormality"])
    thalch = st.number_input("Maximum Heart Rate Achieved (thalch)", min_value=50, max_value=250, value=150)
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=["False", "True"])

# --- 4. PREPROCESS USER INPUT TO MATCH TRAINED COLUMNS ---
if st.button("Predict Heart Disease Risk", type="primary"):
    
    # Map binary features exactly like cell [18]
    sex_encoded = 1 if sex == "Male" else 0
    exang_encoded = 1 if exang == "True" else 0
    fbs_encoded = 1 if fbs == "True" else 0
    
    # Create a DataFrame template matching your `x_train` feature columns exactly
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs_encoded],
        'thalch': [thalch],
        'exang': [exang_encoded],
        'oldpeak': [oldpeak],
        'cp_atypical angina': [1 if cp == "atypical angina" else 0],
        'cp_non-anginal': [1 if cp == "non-anginal" else 0],
        'cp_typical angina': [1 if cp == "typical angina" else 0],
        'restecg_normal': [1 if restecg == "normal" else 0],
        'restecg_st-t abnormality': [1 if restecg == "st-t abnormality" else 0]
    })
    
    # Apply feature scaling to continuous columns if scaler is available
    scale_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    if scaler is not None:
        input_data[scale_cols] = scaler.transform(input_data[scale_cols])
    else:
        # Warning fall-back if you forgot to export the scaler object
        st.warning("Using unscaled raw inputs. For optimal accuracy, export and load your trained StandardScaler.")

    # --- 5. GENERATE INFERENCE AND DISPLAY ---
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.header("Prediction Analysis")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected:** The model predicts **Heart Disease Present** with a probability of **{probability:.2%}**.")
    else:
        st.success(f"✅ **Low Risk:** The model predicts **No Heart Disease** with a probability of **{(1 - probability):.2%}**.")