from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


models = {
    'linear_regression': joblib.load('linear_regression_model.pkl'),
    'polynomial_regression': joblib.load('polynomial_regression_model.pkl'),
    'decision_tree': joblib.load('decision_tree_model.pkl'),
    'random_forest': joblib.load('random_forest_model.pkl')
}

@app.route('/')
def home():
    return "Insurance Charges Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        
        required_fields = ['age', 'bmi', 'children', 'sex', 'smoker', 'region', 'model']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        
        age = int(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        sex = data['sex']
        smoker = data['smoker']
        region = data['region']
        model_name = data['model']

        
        label_mapping = {'male': 0, 'female': 1, 'no': 0, 'yes': 1, 'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        sex = label_mapping.get(sex, -1)
        smoker = label_mapping.get(smoker, -1)
        region = label_mapping.get(region, -1)

        if -1 in [sex, smoker, region]:
            return jsonify({"error": "Invalid categorical input values."}), 400

        
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

        if model_name not in models:
            return jsonify({"error": "Invalid model name."}), 400

        model = models[model_name]
        prediction = model.predict(input_data)

        return jsonify({'predicted_charges': round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
