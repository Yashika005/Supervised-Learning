import streamlit as st
import requests
import json
 
# URL of the Flask API
API_URL = "http://localhost:5000/predict"  # Make sure your Flask app is running
 
# Streamlit UI elements
st.title("Insurance Charges Prediction")
 
st.write("""
    This app predicts insurance charges based on your personal details.
    Fill out the form below and get an estimate of your insurance charges.
""")
 
# Form fields for user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
 
# Button to submit the form and get the prediction
if st.button("Predict Charges"):
    # Prepare the data to send to the API
    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }
 
    # Make POST request to Flask API
    try:
        response = requests.post(API_URL, json=input_data)
        response_data = response.json()
 
        if response.status_code == 200:
            # Display predicted charges
            predicted_charges = response_data.get("predicted_charges")
            st.success(f"Predicted Insurance Charges: ${predicted_charges}")
        else:
            # Handle errors from the API
            st.error(f"Error: {response_data.get('error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
 