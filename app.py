import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64


model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("column.pkl")



# def set_bg_image(image_file):
#     with open(image_file, "rb") as img:
#         encoded = base64.b64encode(img.read()).decode()

#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpg;base64,{encoded}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )


# set_bg_image("Cool-Gym-Wallpaper-HD.jpg")


import streamlit as st

st.markdown("""
<style>

/* ===== Global Input Styling ===== */
input, textarea {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 8px !important;
}

/* ===== Number Input Arrows ===== */
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    opacity: 1;
}

/* ===== Selectbox & Multiselect ===== */
div[data-baseweb="select"] {
    background-color: #0f172a !important;
    border-radius: 8px !important;
    border: 2px solid #38bdf8 !important;
    color: white !important;
}

/* ===== Slider ===== */
div[data-baseweb="slider"] > div > div {
    background: linear-gradient(to right, #38bdf8, #22c55e) !important;
}

div[data-baseweb="slider"] span {
    background-color: #38bdf8 !important;
    border-color: #38bdf8 !important;
}

/* ===== Checkbox & Radio ===== */
div[data-baseweb="checkbox"] span,
div[data-baseweb="radio"] span {
    background-color: #38bdf8 !important;
}

/* ===== Buttons ===== */
.stButton > button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    font-weight: bold;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)



st.title("Heart Disease Risk Prediction ")
age = st.slider("Age", 10, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.slider("BMI", 15.0, 45.0, 25.0)
blood_pressure = st.slider("Blood Pressure", 80, 180, 120)
cholesterol = st.slider("Cholesterol", 100, 350, 200)
blood_sugar = st.slider("Blood Sugar", 50, 300, 100)

workout_intensity = st.selectbox("Workout Intensity", ["Low", "Medium", "High"])
workout_frequency = st.slider("Workout Frequency (days/week)", 0, 7, 3)
workout_duration = st.slider("Workout Duration (minutes)", 0, 200, 60)

smoking = st.selectbox("Smoking", ["No", "Yes"])
family_hd = st.selectbox("Family Heart Disease", ["No", "Yes"])
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
family_hd = 1 if family_hd == "Yes" else 0

stress_map = {"Low": 0, "Medium": 1, "High": 2}
workout_map = {"Low": 0, "Medium": 1, "High": 2}

stress = stress_map[stress]
workout_intensity = workout_map[workout_intensity]
if st.button("Predict Heart Risk"):
      input_data = np.array([[age, gender, bmi, blood_pressure, cholesterol,blood_sugar, workout_intensity, workout_frequency, workout_duration, smoking, family_hd, sleep_hours, stress]])
      df_input = pd.DataFrame(input_data, columns=columns)
      df_scaled = scaler.transform(df_input)
      prediction = model.predict(df_scaled)[0]
      if prediction == 1:
         st.error("⚠️ High Risk of Heart Disease")
      else:
         st.success("✅ Low Risk of Heart Disease")


