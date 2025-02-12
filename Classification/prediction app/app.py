import streamlit as st
import requests

API_URL = "http://localhost:5000/predict"

st.title("Iris Species Classification")



sepal_length = st.text_input("Enter Sepal Length (cm)", "5.1")
sepal_width = st.text_input("Enter Sepal Width (cm)", "3.5")
petal_length = st.text_input("Enter Petal Length (cm)", "1.4")
petal_width = st.text_input("Enter Petal Width (cm)", "0.2")

# Model Selection
selected_model = st.selectbox("Choose Classification Model", ["KNN", "Logistic Regression", "Decision Tree", "Random Forest"])

if st.button("Predict"):
    try:
        input_data = {
            "sepal_length": float(sepal_length),
            "sepal_width": float(sepal_width),
            "petal_length": float(petal_length),
            "petal_width": float(petal_width),
            "model": selected_model,
        }

        response = requests.post(API_URL, json=input_data).json()

        if "predicted_species" in response:
            st.success(f"The predicted species is: {response['predicted_species']}")
        else:
            st.error(f"Error: {response.get('error', 'Unknown error')}")
    except ValueError:
        st.error("Please enter valid numerical values for all input fields.")
