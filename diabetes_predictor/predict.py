import sys
import pandas as pd
import joblib

rc = joblib.load('models/model.pkl')

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict.py', methods=['POST'])
def predict():
    input_data = request.json['input_data']
    # Replace with actual model prediction logic
    prediction = f"Predicted result for {input_data}"
    return jsonify({'prediction': prediction})

gender = int(sys.argv[1])
smoking = int(sys.argv[2])
age = int(sys.argv[3])
hypertension = int(sys.argv[4])
heartdisease = int(sys.argv[5])
bmi = float(sys.argv[6])
Haemoglobin = float(sys.argv[7])
glucose = int(sys.argv[8])
pregnancy = int(sys.argv[9])
insulin = int(sys.argv[10])

testing_data = pd.DataFrame([[age, hypertension, heartdisease, bmi, Haemoglobin, glucose, smoking, gender, pregnancy, insulin]],
                            columns=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'smoking_history_encoded', 'gender_encoded', 'Pregnancies', 'Insulin'])

tested_data = rc.predict_proba(testing_data)[:, 1]
prediction = "Person is diabetic" if tested_data[0] * 100 >= 75 else "Person is not diabetic"

print(prediction, "and the percentage of being diabetic is: ", tested_data[0]*100)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
