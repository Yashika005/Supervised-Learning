# Iris Species Classification

This repository contains an implementation of an **Iris Species Classification** system using multiple machine learning models. The project consists of model training, an API for predictions, and a Streamlit-based web application.

## Project Structure
```
├── train_models.py        # Script to train and save classification models
├── predict.py             # Flask API for making predictions
├── app.py                 # Streamlit UI for user interaction
├── iris.csv               # Dataset used for training
├── models/                # Directory where trained models are stored
└── Classification.ipynb   # Jupyter Notebook with classification workflow
```

## Training the Models
To train the classification models, run:
```sh
python train_models.py
```
This script:
- Loads the Iris dataset (`iris.csv`).
- Trains four different models: **KNN, Logistic Regression, Decision Tree, and Random Forest**.
- Saves the trained models in the `models/` directory.

## Running the API
To start the Flask API, execute:
```sh
python predict.py
```
The API will be available at `http://localhost:5000/`.

### API Endpoints
- **`GET /`**: Health check endpoint.
- **`POST /predict`**: Make a prediction by providing input features.
  - Example request:
    ```json
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2,
      "model": "Random Forest"
    }
    ```
  - Example response:
    ```json
    {
      "predicted_species": "Iris-setosa"
    }
    ```

## Running the Web Application
To launch the Streamlit UI, execute:
```sh
streamlit run app.py
```
This will start a web interface where users can input flower measurements and select a model for prediction.

## Dependencies
The main libraries used in this project are:
- Flask
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

# Insurance Charges Prediction

This repository provides an implementation of an **Insurance Charges Prediction** system using multiple machine learning models. The project includes model training, an API for making predictions, and a Streamlit-based web application.

## Project Structure
```
├── train_models.py         # Script to train and save regression models
├── app.py                  # Flask API for making predictions
├── streamlit_app.py         # Streamlit UI for user interaction
├── insurance.csv           # Dataset used for training
├── models/                 # Directory where trained models are stored
└── Various trained models (.pkl)
```

## Training the Models
To train the regression models, run:
```sh
python train_models.py
```
This script:
- Loads the Insurance dataset (`insurance.csv`).
- Trains four different models: **Linear Regression, Polynomial Regression, Decision Tree, and Random Forest**.
- Saves the trained models in the `models/` directory.

## Running the API
To start the Flask API, execute:
```sh
python app.py
```
The API will be available at `http://localhost:5000/`.

### API Endpoints
- **`GET /`**: Health check endpoint.
- **`POST /predict`**: Make a prediction by providing input features.
  - Example request:
    ```json
    {
      "age": 30,
      "bmi": 25.0,
      "children": 0,
      "sex": "male",
      "smoker": "no",
      "region": "northeast",
      "model": "random_forest"
    }
    ```
  - Example response:
    ```json
    {
      "predicted_charges": 12000.50
    }
    ```

## Running the Web Application
To launch the Streamlit UI, execute:
```sh
streamlit run streamlit_app.py
```
This will start a web interface where users can input their details and select a model for prediction.

## Dependencies
The main libraries used in this project are:
- Flask
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

