import streamlit as st
import numpy as np
import pickle
from model import LogisticRegressionScratch

# -------------------- Load model and scaler (UNCHANGED) --------------------
model = pickle.load(open("cardio_model.pkl", "rb"))
X_min, X_max = pickle.load(open("scaler.pkl", "rb"))

# -------------------- Page Config (UI) --------------------
st.set_page_config(
    page_title="Cardio Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -------------------- Custom CSS (UI ONLY) --------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #d62828;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background-color: #f8f9fa;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.result-good {
    color: #2d6a4f;
    font-size: 26px;
    font-weight: bold;
}
.result-bad {
    color: #d62828;
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Title Section --------------------
st.markdown('<div class="main-title">❤️ Cardiovascular Disease Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered health risk assessment using Machine Learning</div>',
    unsafe_allow_html=True
)

# -------------------- Sidebar UI --------------------
st.sidebar.markdown("## 🧑 Patient Information")

age = st.sidebar.number_input("Age (Years)", 18, 100, 30)
gender = st.sidebar.selectbox(
    "Gender", [1, 2],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

st.sidebar.markdown("### 📏 Body Measurements")
height = st.sidebar.number_input("Height (cm)", 120, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)

st.sidebar.markdown("### 🩺 Blood Pressure")
ap_hi = st.sidebar.number_input("Systolic BP", 80, 250, 120)
ap_lo = st.sidebar.number_input("Diastolic BP", 60, 200, 80)

st.sidebar.markdown("### 🧪 Medical Levels")
cholesterol = st.sidebar.selectbox(
    "Cholesterol",
    [1, 2, 3],
    format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1]
)

gluc = st.sidebar.selectbox(
    "Glucose",
    [1, 2, 3],
    format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1]
)

st.sidebar.markdown("### 🚭 Lifestyle Habits")
smoke = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alco = st.sidebar.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
active = st.sidebar.selectbox("Physically Active", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Risk")

# -------------------- Prediction Section (LOGIC UNCHANGED) --------------------
if predict_btn:
    values = np.array([
        age, gender, height, weight,
        ap_hi, ap_lo, cholesterol,
        gluc, smoke, alco, active
    ]).reshape(1, -1)

    # Normalize input (UNCHANGED)
    values = (values - X_min) / (X_max - X_min)

    prob = model.predict_proba(values)[0]
    prediction = "Disease Detected" if prob >= 0.5 else "No Disease"

    # -------------------- Result UI --------------------
    st.markdown("## 📊 Prediction Result")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if prob >= 0.5:
                st.markdown(f"<div class='result-bad'>⚠ {prediction}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-good'>✅ {prediction}</div>", unsafe_allow_html=True)

            st.markdown(f"### 🔢 Probability: **{round(prob * 100, 2)}%**")

        with col2:
            st.markdown("### 📈 Risk Comparison")
            st.bar_chart({
                "No Disease": [1 - prob],
                "Disease": [prob]
            })

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "<center>💙 Built with Streamlit & Machine Learning | Educational Purpose Only</center>",
    unsafe_allow_html=True
)
