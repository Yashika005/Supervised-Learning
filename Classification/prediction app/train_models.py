import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


iris_df = pd.read_csv("iris.csv")


label_encoder = LabelEncoder()
iris_df["species"] = label_encoder.fit_transform(iris_df["species"])


X = iris_df.drop(columns=["species"])
y = iris_df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
}


os.makedirs("models", exist_ok=True)

trained_models = {}
accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)
    
   
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}_model.pkl")


joblib.dump(label_encoder, "models/label_encoder.pkl")


accuracy_df = pd.DataFrame(accuracies.items(), columns=["Model", "Accuracy"])
accuracy_df.to_csv("models/model_accuracies.csv", index=False)

print("Training completed. Models saved successfully!")
