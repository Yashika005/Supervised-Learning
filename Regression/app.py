from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def home():
    return "Insurance Charges Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Inside /predict method")

        data = request.get_json()

        required_fields = ['age', 'bmi', 'children', 'sex', 'smoker', 'region']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        age = float(data.get('age'))
        bmi = float(data.get('bmi'))
        children = int(data.get('children'))
        sex = data.get('sex')
        smoker = data.get('smoker')
        region = data.get('region')

        print(f"{age},{bmi},{children}, {sex}, {smoker}, {region}")

#         return jsonify({'predicted_charges': age})


        label_mapping = {'male': 0, 'female': 1, 'no': 0, 'yes': 1, 'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        
        sex = label_mapping.get(sex, -1)
        smoker = label_mapping.get(smoker, -1)
        region = label_mapping.get(region, -1)

        if -1 in [sex, smoker, region]:
            return jsonify({"error": "Invalid input for categorical variables."}), 400

        
        input_data = pd.DataFrame([[age,sex, bmi, children,  smoker, region]],
                                  columns=['age','sex', 'bmi', 'children', 'smoker', 'region'])

        prediction = model.predict(input_data)

        return jsonify({'predicted_charges': round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
