import streamlit as st
import joblib
import os

def prepare_features(
    gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
    fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath,
    swallowing_difficulty, chest_pain
):
    gender = 1 if gender == "Male" else 0
    return [[
        gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
        fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath,
        swallowing_difficulty, chest_pain
    ]]

st.title("Lung Cancer Prediction")

gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=1, max_value=120, value=30)

binary_options = {"No": 1, "Yes": 2}
smoking = st.selectbox("Do you smoke?", options=["No", "Yes"])
yellow_fingers = st.selectbox("Do you have yellow fingers?", options=["No", "Yes"])
anxiety = st.selectbox("Do you experience anxiety?", options=["No", "Yes"])
peer_pressure = st.selectbox("Are you under peer pressure?", options=["No", "Yes"])
chronic_disease = st.selectbox("Do you have any chronic disease?", options=["No", "Yes"])
fatigue = st.selectbox("Do you experience fatigue?", options=["No", "Yes"])
allergy = st.selectbox("Do you have allergies?", options=["No", "Yes"])
wheezing = st.selectbox("Do you experience wheezing?", options=["No", "Yes"])
alcohol_consuming = st.selectbox("Do you consume alcohol?", options=["No", "Yes"])
coughing = st.selectbox("Do you cough frequently?", options=["No", "Yes"])
shortness_of_breath = st.selectbox("Do you experience shortness of breath?", options=["No", "Yes"])
swallowing_difficulty = st.selectbox("Do you have swallowing difficulty?", options=["No", "Yes"])
chest_pain = st.selectbox("Do you have chest pain?", options=["No", "Yes"])

features = prepare_features(
    gender, age,
    binary_options[smoking], binary_options[yellow_fingers], binary_options[anxiety],
    binary_options[peer_pressure], binary_options[chronic_disease], binary_options[fatigue],
    binary_options[allergy], binary_options[wheezing], binary_options[alcohol_consuming],
    binary_options[coughing], binary_options[shortness_of_breath],
    binary_options[swallowing_difficulty], binary_options[chest_pain]
)

model = joblib.load("D:\FREELANCE_PROJECTS\lung-cancer-prediction\models\lung_cancer_model.pkl")

if st.button("Predict"):
    prediction = model.predict(features)
    result = "Positive for Lung Cancer" if prediction[0] == 1 else "Negative for Lung Cancer"
    st.success(result)
