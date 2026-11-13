import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("students.csv")

X = data[["hours", "attendance", "marks"]]
y = data["performance"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")