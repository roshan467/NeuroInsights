# src/api.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# === App & Model Paths ===
app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.joblib')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/processed/merged_new_dataset.csv')

# === Load Model & Data ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Merged dataset not found at {DATA_PATH}")

model = joblib.load(MODEL_PATH)
merged = pd.read_csv(DATA_PATH)

# === Prediction Endpoint ===
@app.route('/predict/<patient_id>', methods=['GET'])
def predict(patient_id):
    patient = merged[merged['patient_id']==patient_id]
    if patient.empty:
        return jsonify({"error": "Patient not found"}), 404
    
    # Features used for prediction
    feature_cols = ['alpha_power', 'beta_power', 'gamma_power', 'theta_power', 
                    'mean_intensity', 'variance']
    missing_cols = [c for c in feature_cols if c not in patient.columns]
    if missing_cols:
        return jsonify({"error": f"Missing columns for prediction: {missing_cols}"}), 400
    
    X = patient[feature_cols]
    pred = model.predict(X)
    probs = model.predict_proba(X).tolist()[0] if hasattr(model, "predict_proba") else None
    
    return jsonify({
        "patient_id": patient_id,
        "prediction": int(pred[0]),
        "probabilities": probs
    })

# === Optional: List all patients ===
@app.route('/patients', methods=['GET'])
def list_patients():
    return jsonify({"patient_ids": merged['patient_id'].tolist()})

# === Run Flask App ===
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
