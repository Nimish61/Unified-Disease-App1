import json
import os
import urllib.error
import urllib.request
from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = base_dir

load_errors = {}


def load_model(filename):
    try:
        return joblib.load(os.path.join(models_dir, filename))
    except Exception as e:
        load_errors[filename] = str(e)
        return None


models = {
    'diabetes': load_model('Diabetes_best_model.joblib'),
    'heart_failure': load_model('Heart_Failure_best_model.joblib'),
    'kidney': load_model('CKD_best_model.joblib'),
    'heart': load_model('Heart_best_model.joblib'),
    'stroke': load_model('Stroke_best_model.joblib'),
    'liver': load_model('Liver_best_model.joblib'),
}

# -------------------------------------------------------------------------
# Clinical Feature Metadata: Weights, Defaults (Medians), & Critical Flags
# -------------------------------------------------------------------------
DISEASE_METADATA = {
    'Diabetes': {
        'model_key': 'diabetes',
        'critical_features': ['Glucose'],
        'features': {
            'Glucose': {'weight': 0.35, 'default': 117.0, 'payload_key': 'Glucose'},
            'BMI': {'weight': 0.20, 'default': 32.0, 'payload_key': 'BMI'},
            'Age': {'weight': 0.15, 'default': 33.0, 'payload_key': 'Age'},
            'DiabetesPedigreeFunction': {'weight': 0.10, 'default': 0.3725, 'payload_key': 'DiabetesPedigreeFunction'},
            'BloodPressure': {'weight': 0.08, 'default': 72.0, 'payload_key': 'BloodPressure'},
            'Pregnancies': {'weight': 0.05, 'default': 3.0, 'payload_key': 'Pregnancies'},
            'Insulin': {'weight': 0.04, 'default': 30.5, 'payload_key': 'Insulin'},
            'SkinThickness': {'weight': 0.03, 'default': 23.0, 'payload_key': 'SkinThickness'},
        },
    },
    'Heart Failure': {
        'model_key': 'heart_failure',
        'critical_features': ['ejection_fraction', 'serum_creatinine'],
        'features': {
            'time': {'weight': 0.35, 'default': 250.0, 'payload_key': 'FollowUpTime'}, # Changed to safe 250
            'serum_creatinine': {'weight': 0.22, 'default': 1.0, 'payload_key': 'SerumCreatinine'},
            'ejection_fraction': {'weight': 0.20, 'default': 55.0, 'payload_key': 'EjectionFraction'}, # Changed to safe 55
            'age': {'weight': 0.08, 'default': 55.0, 'payload_key': 'Age'},
            'serum_sodium': {'weight': 0.05, 'default': 140.0, 'payload_key': 'SerumSodium'},
            'creatinine_phosphokinase': {'weight': 0.03, 'default': 150.0, 'payload_key': 'CreatininePhosphokinase'},
            'platelets': {'weight': 0.03, 'default': 260000.0, 'payload_key': 'Platelets'},
            'high_blood_pressure': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Hypertension'},
            'anaemia': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Anaemia'},
            'sex': {'weight': 0.01, 'default': 1.0, 'payload_key': 'Gender'},
            'smoking': {'weight': 0.005, 'default': 0.0, 'payload_key': 'Smoking'},
            'diabetes': {'weight': 0.005, 'default': 0.0, 'payload_key': 'DiabetesMellitus'},
        },
    },
    'Kidney Disease': {
        'model_key': 'kidney',
        'critical_features': ['SerumCreatinine', 'SpecificGravity'],
        'features': {
            'SpecificGravity': {'weight': 0.25, 'default': 1.02, 'payload_key': 'SpecificGravity'},
            'SerumCreatinine': {'weight': 0.22, 'default': 1.0, 'payload_key': 'SerumCreatinine'},
            'RedBloodCells': {'weight': 0.15, 'default': 0.0, 'payload_key': 'RedBloodCells'},
            'Albumin': {'weight': 0.12, 'default': 0.0, 'payload_key': 'Albumin'},
            'PackedCellVolume': {'weight': 0.08, 'default': 40.0, 'payload_key': 'PackedCellVolume'},
            'Hemoglobin': {'weight': 0.06, 'default': 12.5, 'payload_key': 'Hemoglobin'},
            'Hypertension': {'weight': 0.04, 'default': 0.0, 'payload_key': 'Hypertension'},
            'DiabetesMellitus': {'weight': 0.03, 'default': 0.0, 'payload_key': 'DiabetesMellitus'},
            'BloodGlucoseRandom': {'weight': 0.02, 'default': 121.0, 'payload_key': 'BloodGlucoseRandom'},
            'BloodUrea': {'weight': 0.01, 'default': 42.0, 'payload_key': 'BloodUrea'},
            'Sodium': {'weight': 0.01, 'default': 138.0, 'payload_key': 'SerumSodium'},
            'Potassium': {'weight': 0.01, 'default': 4.4, 'payload_key': 'Potassium'},
        },
    },
    'Heart Disease': {
        'model_key': 'heart',
        'critical_features': ['Thalassemia', 'ChestPainType'],
        'features': {
            'Thalassemia': {'weight': 0.28, 'default': 2.0, 'payload_key': 'Thalassemia'},
            'NumMajorVessels': {'weight': 0.22, 'default': 0.0, 'payload_key': 'NumMajorVessels'},
            'ChestPainType': {'weight': 0.18, 'default': 1.0, 'payload_key': 'ChestPainType'},
            'MaxHeartRate': {'weight': 0.12, 'default': 153.0, 'payload_key': 'MaxHeartRate'},
            'Oldpeak': {'weight': 0.08, 'default': 0.8, 'payload_key': 'Oldpeak'},
            'Age': {'weight': 0.04, 'default': 55.0, 'payload_key': 'Age'},
            'ExerciseAngina': {'weight': 0.03, 'default': 0.0, 'payload_key': 'ExerciseAngina'},
            'RestingBP': {'weight': 0.02, 'default': 130.0, 'payload_key': 'BloodPressure'},
            'Slope': {'weight': 0.015, 'default': 1.0, 'payload_key': 'Slope'},
            'Cholesterol': {'weight': 0.01, 'default': 240.0, 'payload_key': 'Cholesterol'},
            'Sex': {'weight': 0.005, 'default': 1.0, 'payload_key': 'Gender'},
        },
    },
    'Stroke': {
        'model_key': 'stroke',
        'critical_features': ['age'],
        'features': {
            'age': {'weight': 0.45, 'default': 45.0, 'payload_key': 'Age'},
            'avg_glucose_level': {'weight': 0.25, 'default': 106.0, 'payload_key': 'Glucose'},
            'bmi': {'weight': 0.15, 'default': 28.1, 'payload_key': 'BMI'},
            'hypertension': {'weight': 0.05, 'default': 0.0, 'payload_key': 'Hypertension'},
            'heart_disease': {'weight': 0.04, 'default': 0.0, 'payload_key': 'CoronaryArteryDisease'},
            'smoking_status': {'weight': 0.03, 'default': 0.0, 'payload_key': 'Smoking'},
            'work_type': {'weight': 0.015, 'default': 2.0, 'payload_key': 'WorkType'},
            'Residence_type': {'weight': 0.01, 'default': 1.0, 'payload_key': 'ResidenceType'},
            'gender': {'weight': 0.005, 'default': 1.0, 'payload_key': 'Gender'},
        },
    },
    'Liver Disease': {
        'model_key': 'liver',
        'positive_class_index': 0,  # FIX: ILPD dataset encodes disease as class 1 (index 0)
        'critical_features': [
            'Direct_bilirubin',
            'Alamine_Aminotransferase',
        ],
        'features': {
            'Direct_bilirubin': {'weight': 0.26, 'default': 0.3, 'payload_key': 'DirectBilirubin'},
            'Alkaline_Phosphotase': {'weight': 0.22, 'default': 208.0, 'payload_key': 'AlkalinePhosphotase'},
            'Alamine_Aminotransferase': {'weight': 0.20, 'default': 35.0, 'payload_key': 'AlamineAminotransferase'},
            'Aspartate_Aminotransferase': {'weight': 0.12, 'default': 42.0, 'payload_key': 'AspartateAminotransferase'},
            'Total_bilirubin': {'weight': 0.08, 'default': 1.0, 'payload_key': 'TotalBilirubin'},
            'age': {'weight': 0.05, 'default': 45.0, 'payload_key': 'Age'},
            'Total_Proteins': {'weight': 0.03, 'default': 6.6, 'payload_key': 'TotalProteins'},
            'Albumin': {'weight': 0.02, 'default': 3.1, 'payload_key': 'Albumin'},
            'Albumin_and_Globulin_Ratio': {'weight': 0.015, 'default': 0.93, 'payload_key': 'AlbuminGlobulinRatio'},
            'gender': {'weight': 0.005, 'default': 1.0, 'payload_key': 'Gender'},
        },
    },
}


def generate_elaborate_6month_routine(evaluated_diseases, patient_vitals, alert_biomarkers):
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        return 'OpenRouter API key is not configured.'

    prompt = f"""
    You are an expert Clinical AI and Preventative Medicine Consultant.
    Based strictly on the machine learning risk predictions and verified patient biomarkers below, construct a rigorous, structured 6-Month Preventative Routine.

    PATIENT CLINICAL BASELINE:
    - Age: {patient_vitals.get('Age', 'Not specified')}
    - BMI: {patient_vitals.get('BMI', 'Not specified')}
    - Resting Blood Pressure: {patient_vitals.get('BloodPressure', 'Not specified')} mmHg
    - Blood Glucose: {patient_vitals.get('Glucose', 'Not specified')} mg/dL

    EVALUATED PREDICTION PROBABILITIES & CONFIDENCE SCORES:
    {json.dumps(evaluated_diseases, indent=2)}

    ABNORMAL BIOMARKERS DRIVING ELEVATED RISK:
    {json.dumps(alert_biomarkers, indent=2) if alert_biomarkers else 'Biomarkers within standard reference ranges.'}

    REQUIREMENTS:
    1. Structure the response strictly in chronological phases:
       - Phase 1: Months 1-2 (Acute Stabilization & Metabolic Deceleration)
       - Phase 2: Months 3-4 (Vascular Remodeling & Systemic Reconditioning)
       - Phase 3: Months 5-6 (Long-term Maintenance & Re-Testing Benchmarks)
    2. Under each Phase, provide:
       - <strong>Targeted Nutrition & Micronutrient Constraints</strong> (specify daily limits, e.g., sodium mg, glycemic index).
       - <strong>Prescribed Physical Regimen</strong> (type, target heart rate reserve %, weekly frequency, contraindications).
       - <strong>Direct Biomarker Impact</strong> (state explicitly which failing metric this phase stabilizes).
    3. Include a final section: <strong>Critical Clinical Retesting Schedule</strong> (explicit lab panels to draw at Month 3 and Month 6).
    4. NO medical prescription of pharmaceutical drugs. Keep tone authoritative, clinical, and precise.
    5. Output CLEAN HTML only using <h3>, <h4>, <p>, <ul>, <li> tags. Do not wrap in markdown code fences.
    """

    payload = {
        'model': 'openrouter/free',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.2,
        'max_tokens': 950,
    }

    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://unified-disease-app1.vercel.app',
            'X-Title': 'VitaPredict Clinical System',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=9) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            if 'choices' in res_data and len(res_data['choices']) > 0:
                return res_data['choices'][0]['message']['content']
            return f"<p style='color: #d9534f;'>Generation failed: {json.dumps(res_data)}</p>"
    except Exception as e:
        return f"<p style='color: #d9534f;'>Routine generation error: {str(e.__class__.__name__)} - {str(e)}</p>"


@app.route('/api/predict/general', methods=['POST'])
def predict_general():
    if load_errors:
        return jsonify({'status': 'error', 'message': f'Model loading failure: {load_errors}'}), 500

    data = request.json or {}
    evaluated_diseases = {}
    biomarker_alerts = {}

    for disease_name, meta in DISEASE_METADATA.items():
        model = models.get(meta['model_key'])
        if not model:
            continue

        # 1. Check if critical/primary biomarkers exist
        # 1. Reject if critical features are missing
    missing_critical = False
    for crit in meta['critical_features']:
        payload_key = meta['features'][crit]['payload_key']
        val = data.get(payload_key)
        if val in [None, '', 'null']:
            missing_critical = True
            break

    if missing_critical:
        continue  # Completely skips evaluating this disease

    # 2. Build row, apply safe defaults, and calculate confidence
    feature_row = {}
    present_weight = 0.0

    for feat_name, feat_meta in meta['features'].items():
        val = data.get(feat_meta['payload_key'])
        if val not in [None, '', 'null']:
            try:
                parsed_val = float(val)
                feature_row[feat_name] = parsed_val
                present_weight += feat_meta['weight'] # Boosts confidence score
            except ValueError:
                feature_row[feat_name] = feat_meta['default']
        else:
            # Applies safe default, does NOT add to present_weight
            feature_row[feat_name] = feat_meta['default']

    confidence_score = round(min(present_weight * 100, 100.0), 1)

    # 3. Reject if overall data quality is too low
    if confidence_score < 65.0:
        continue  # Skips predicting if > 35% of model importance is missing
        # 3. Generate inference
        try:
            df_input = pd.DataFrame([feature_row])
            
            # FIX: Pull the correct probability index based on dataset labeling
            pos_idx = meta.get('positive_class_index', 1) # Defaults to 1 for normal datasets
            pred_prob = round(float(model.predict_proba(df_input)[0][pos_idx]) * 100, 2)

            evaluated_diseases[disease_name] = {
                'probability': pred_prob,
                'confidence': confidence_score,
                'status': ('High Risk' if pred_prob >= 50.0 else ('Moderate Risk' if pred_prob >= 20.0 else 'Low Risk')),
            }
        except Exception as e:
            continue

    if not evaluated_diseases:
        return jsonify({
            'status': 'error',
            'message': 'Insufficient clinical parameters provided. Please input primary diagnostic markers (e.g., Glucose, Blood Pressure, Age).'
        })

    # Generate phased 6-month clinical routine based on vetted results
    ai_routine = generate_elaborate_6month_routine(evaluated_diseases, data, biomarker_alerts)

    return jsonify({
        'status': 'success',
        'predictions': evaluated_diseases,
        'biomarker_alerts': biomarker_alerts,
        'routine': ai_routine,
    })


if __name__ == '__main__':
    app.run(debug=True)
