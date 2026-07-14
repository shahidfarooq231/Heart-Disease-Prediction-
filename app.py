"""
Heart Disease Prediction - Streamlit Demo App
Run with:
    streamlit run app/app.py
"""

import os
import json

import joblib
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")
METADATA_PATH = os.path.join(ROOT, "models", "metadata.json")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, scaler, metadata


st.title(" Heart Disease Risk Predictor")
st.caption("Machine Learning project — Edunet Foundation / IBM SkillsBuild Internship")

try:
    model, scaler, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please run `python src/train.py` first to train and save the model.")
    st.stop()

st.markdown(
    f"**Model in use:** {metadata['best_model']} &nbsp;|&nbsp; "
    f"**Test Accuracy:** {metadata['accuracy']*100:.1f}% &nbsp;|&nbsp; "
    f"**ROC-AUC:** {metadata['roc_auc']:.3f}"
)

st.divider()
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox(
        "Chest Pain Type", options=[0, 1, 2, 3],
        format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x]
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox(
        "Resting ECG Result", options=[0, 1, 2],
        format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x]
    )

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", options=[0, 1, 2],
        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]
    )
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia", options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x]
    )

st.divider()

if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f" High risk of heart disease detected (probability: {probability*100:.1f}%)")
    else:
        st.success(f" Low risk of heart disease (probability: {probability*100:.1f}%)")

    st.progress(min(int(probability * 100), 100))
    st.caption(
        "Disclaimer: This tool is for educational purposes only as part of an academic "
        "internship project. It is not a substitute for professional medical diagnosis."
    )

st.divider()
st.caption("Built with scikit-learn + Streamlit | Dataset: UCI Heart Disease (Cleveland)")
