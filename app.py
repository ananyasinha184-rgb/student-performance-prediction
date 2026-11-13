
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load existing ML model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    # Read student dataset
    df = pd.read_csv("students.csv")

    # Convert columns into numbers
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
    df["marks"] = pd.to_numeric(df["marks"], errors="coerce")

    # Remove incomplete records
    df = df.dropna(
        subset=["hours", "attendance", "marks", "performance"]
    )

    # Dashboard statistics
    total_students = len(df)
    average_marks = round(df["marks"].mean(), 2)
    average_attendance = round(df["attendance"].mean(), 2)

    # Performance distribution
    performance = df["performance"].astype(str).value_counts()
    performance_labels = performance.index.tolist()
    performance_values = performance.values.tolist()

    # Average marks by category
    grouped = df.groupby("performance")["marks"].mean()
    marks_labels = grouped.index.astype(str).tolist()
    marks_values = grouped.round(2).tolist()

    # Study hours vs marks
    study_points = df[["hours", "marks"]].to_dict(
        orient="records"
    )

    # Prediction
    prediction = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        marks = float(request.form["marks"])

        result = model.predict(pd.DataFrame(
      [[hours, attendance, marks]],
       columns=["hours", "attendance", "marks"]
        )
        )
        prediction = result[0]

    return render_template(
        "index.html",
        prediction=prediction,
        total_students=total_students,
        average_marks=average_marks,
        average_attendance=average_attendance,
        performance_labels=performance_labels,
        performance_values=performance_values,
        marks_labels=marks_labels,
        marks_values=marks_values,
        study_points=study_points
    )


if __name__ == "__main__":
    app.run(debug=True)
