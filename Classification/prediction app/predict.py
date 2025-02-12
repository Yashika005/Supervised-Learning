from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)


model_files = ["knn_model.pkl", "logistic_regression_model.pkl", "decision_tree_model.pkl", "random_forest_model.pkl"]
if not all(os.path.exists(f"models/{file}") for file in model_files):
    raise FileNotFoundError("Model files not found!")

knn_model = joblib.load("models/knn_model.pkl")
logistic_model = joblib.load("models/logistic_regression_model.pkl")
decision_tree_model = joblib.load("models/decision_tree_model.pkl")
random_forest_model = joblib.load("models/random_forest_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


accuracy_df = pd.read_csv("models/model_accuracies.csv")
accuracies = dict(zip(accuracy_df["Model"], accuracy_df["Accuracy"]))


models = {
    "KNN": knn_model,
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
}

@app.route("/")
def home():
    return "Iris Classification API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        
        required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width", "model"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
        model_name = data["model"]

       
        if model_name not in models:
            return jsonify({"error": "Invalid model name. Choose from: KNN, Logistic Regression, Decision Tree, Random Forest"}), 400

     
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

       
        model = models[model_name]
        prediction = model.predict(input_data)
        predicted_species = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"predicted_species": predicted_species})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
