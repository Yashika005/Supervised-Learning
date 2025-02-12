import streamlit as st
import requests
import json

API_URL = "http://localhost:5000/predict"

st.title("Insurance Charges Prediction")


age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
model = st.selectbox("Choose Model", ["linear_regression", "polynomial_regression", "decision_tree", "random_forest"])

if st.button("Predict Charges"):
    input_data = {"age": age, "bmi": bmi, "children": children, "sex": sex, "smoker": smoker, "region": region, "model": model}
    response = requests.post(API_URL, json=input_data).json()
    st.success(f"Predicted Insurance Charges: ${response.get('predicted_charges')}")
