from flask import Flask, render_template, request
import joblib
import numpy as np
import pytesseract
from PIL import Image
import re

# 🔹 Set tesseract path (Windows)
import os

TESSERACT_PATH = os.environ.get("TESSERACT_PATH", "tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")


# 🔹 Strong numeric extraction (handles units, commas, noise)
def extract_value(label, text):
    pattern = rf"{label}.*?(\d+[\.,]?\d*)"
    match = re.search(pattern, text, re.I)
    if match:
        return float(match.group(1).replace(",", "."))
    return 0.0


# 🔹 Patient name extraction
def extract_name(text):
    patterns = [
        r"Patient Name\s*[:\-]?\s*([A-Za-z ]+)",
        r"Name\s*[:\-]?\s*([A-Za-z ]+)",
        r"Patient\s*[:\-]?\s*([A-Za-z ]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).strip()
    return "Unknown"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "report" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["report"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")

    try:
        img = Image.open(file)

        # OCR
        text = pytesseract.image_to_string(img)

        # 🔹 Extract name
        patient_name = extract_name(text)

        # 🔹 Extract medical values
        pregnancies = extract_value("Pregnancies", text)
        glucose = extract_value("Glucose", text)
        bp = extract_value("Blood Pressure", text)
        skin = extract_value("Skin Thickness", text)
        insulin = extract_value("Insulin", text)
        bmi = extract_value("BMI", text)
        dpf = extract_value("Pedigree", text)
        age = extract_value("Age", text)

        # 🔹 Prepare ML input
        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        # 🔥 MEDICAL SAFETY OVERRIDE (CRITICAL)
        if glucose >= 200 or bmi >= 30:
            prediction = 1

        # 🔹 Final decision
        if prediction == 1:
            status = "Diabetic"
            condition = "⚠️ High risk of diabetes detected based on the medical report."
            advice = (
                "Please consult a certified doctor immediately for confirmation, "
                "blood tests, and appropriate medical treatment."
            )
        else:
            status = "Non-Diabetic"
            condition = "✅ Low risk of diabetes detected based on the medical report."
            advice = (
                "Maintain a healthy lifestyle, balanced diet, and regular medical checkups."
            )

        return render_template(
            "index.html",
            patient_name=patient_name,
            status=status,
            condition=condition,
            advice=advice,
            glucose=glucose,
            bmi=bmi
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
