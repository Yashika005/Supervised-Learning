import joblib
import pandas as pd

file_path=r"C:\Users\Yashika\Desktop\Inadev\Supervised learning\Regression\linear_regression_model.pkl"
model = joblib.load(file_path)

age = int(23)
bmi = float(25.3)
children = int(2)
sex = 'male'
smoker = 'no'
region = 'northwest'

print(f"{age},{bmi},{children}, {sex}, {smoker}, {region}")

#         return jsonify({'predicted_charges': age})


label_mapping = {'male': 0, 'female': 1, 'no': 0, 'yes': 1, 'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}

sex = label_mapping.get(sex, -1)
smoker = label_mapping.get(smoker, -1)
region = label_mapping.get(region, -1)

print(f"sex {sex}, smoker-{smoker} , region - {region}")


input_data = pd.DataFrame([[age,sex, bmi, children,  smoker, region]],
                                  columns=['age','sex', 'bmi', 'children', 'smoker', 'region'])

result=model.predict(input_data)
print(result)