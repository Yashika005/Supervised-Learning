import requests


url = "http://127.0.0.1:5000/predict"


data = {
    "age": 30,
    "bmi": 25.3,
    "children": 2,
    "sex": "female",
    "smoker": "no",
    "region": "northeast"
}


response = requests.post(url, json=data)


print(response.json())
