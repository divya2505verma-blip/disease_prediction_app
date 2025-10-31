import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="🧬 Disease Prediction App", layout="wide", page_icon="💉")

# -------------------- Custom CSS for Styling --------------------
st.markdown("""
<style>
    /* 🌸 App Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #F8F9FF 0%, #EDE7F6 100%);
        padding: 2rem;
    }

    /* 🪄 Main Title */
    h1 {
        color: #4B3F72;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    /* 🎴 Card Styling */
    [data-testid="stVerticalBlock"] {
        background-color: #FFFFFF;
        border-radius: 18px;
        box-shadow: 0px 6px 16px rgba(108, 99, 255, 0.1);
        padding: 25px;
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stVerticalBlock"]:hover {
        transform: translateY(-3px);
        box-shadow: 0px 8px 20px rgba(108, 99, 255, 0.2);
    }

    /* 🟣 Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #6C63FF, #A892FF);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-size: 16px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: 0.3s ease-in-out;
        box-shadow: 0px 4px 10px rgba(108, 99, 255, 0.3);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 12px rgba(108, 99, 255, 0.5);
    }

    /* 💬 Result Boxes */
    .stAlert {
        border-radius: 12px;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }

    /* 🩸 Subheaders */
    h2, h3 {
        color: #4B3F72;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }

    /* ✨ Footer */
    footer {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        color: #5A5A89;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Models --------------------
diabetes_model = pickle.load(open("disease_models/diabetes_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("disease_models/diabetes_scaler.pkl", "rb"))
heart_model = pickle.load(open("disease_models/heart_model.pkl", "rb"))
heart_scaler = pickle.load(open("disease_models/heart_scaler.pkl", "rb"))

st.markdown("### 🔍 Select the type of disease test you want to perform:")
choice = st.selectbox("Select Option", ["Diabetes", "Heart Disease"])

# =================================================================
# 🩸 DIABETES PREDICTION
# =================================================================
if choice == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose Level", 0, 300)
        BloodPressure = st.number_input("Blood Pressure", 0, 200)
    with col2:
        SkinThickness = st.number_input("Skin Thickness", 0, 100)
        Insulin = st.number_input("Insulin Level", 0, 900)
        BMI = st.number_input("BMI", 0.0, 70.0)
    with col3:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
        Age = st.number_input("Age", 1, 120)

    if st.button("🔍 Predict Diabetes"):
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DiabetesPedigreeFunction, Age]])
        scaled = diabetes_scaler.transform(features)
        prediction = diabetes_model.predict(scaled)
        prob = diabetes_model.predict_proba(scaled)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ High Risk: You may have **Diabetes**.\n\n**Risk Probability:** {prob:.2f}")
        else:
            st.success(f"💚 Great News! You are *Healthy!* \n\n**Risk Probability:** {prob:.2f}")

# =================================================================
# ❤️ HEART DISEASE PREDICTION
# =================================================================
else:
    st.header("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex == "Male" else 0
        cp = st.number_input("Chest Pain Type (0–3)", 0, 3)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200)

    with col2:
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        fbs = 1 if fbs == "True" else 0
        restecg = st.number_input("Rest ECG Results (0–2)", 0, 2)
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250)

    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0
        oldpeak = st.number_input("ST Depression", 0.0, 10.0)
        slope = st.number_input("Slope (0–2)", 0, 2)
        ca = st.number_input("Major Vessels Colored (0–3)", 0, 3)
        thal = st.number_input("Thalassemia (0–3)", 0, 3)

    if st.button("🫀 Predict Heart Disease"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])

        scaled = heart_scaler.transform(features)
        prediction = heart_model.predict(scaled)

        # reverse logic if model predicts opposite
        prediction = 1 - prediction

        if prediction[0] == 1:
            st.error("⚠️ You may have *Heart Disease*. Please consult a doctor.")
        else:
            st.success("💖 Your heart is *Healthy*! Stay fit and keep smiling!")

# -------------------- Footer --------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("✨ Made with ❤️ by **Divya Verma** — College Project")
