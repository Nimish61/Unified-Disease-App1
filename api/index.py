import json
import os
import urllib.error
import urllib.request
from flask import Flask, jsonify, request
import joblib
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
# Full Clinical Metadata (Exact Scikit-Learn Feature Order & Safe Defaults)
# -------------------------------------------------------------------------
DISEASE_METADATA = {
    'Diabetes': {
        'model_key': 'diabetes',
        'critical_features': ['Glucose'],
        'features': {
            'Pregnancies': {'weight': 0.05, 'default': 1.0, 'payload_key': 'Pregnancies'},
            'Glucose': {'weight': 0.35, 'default': 90.0, 'payload_key': 'Glucose'},
            'BloodPressure': {'weight': 0.08, 'default': 80.0, 'payload_key': 'BloodPressure'},
            'SkinThickness': {'weight': 0.03, 'default': 20.0, 'payload_key': 'SkinThickness'},
            'Insulin': {'weight': 0.04, 'default': 15.0, 'payload_key': 'Insulin'},
            'BMI': {'weight': 0.20, 'default': 22.0, 'payload_key': 'BMI'},
            'DiabetesPedigreeFunction': {'weight': 0.10, 'default': 0.25, 'payload_key': 'DiabetesPedigreeFunction'},
            'Age': {'weight': 0.15, 'default': 33.0, 'payload_key': 'Age'}
        }
    },
    'Heart Failure': {
        'model_key': 'heart_failure',
        'critical_features': ['ejection_fraction', 'serum_creatinine', 'time'],
        'features': {
            'age': {'weight': 0.08, 'default': 55.0, 'payload_key': 'Age'},
            'anaemia': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Anaemia'},
            'creatinine_phosphokinase': {'weight': 0.03, 'default': 120.0, 'payload_key': 'CreatininePhosphokinase'},
            'diabetes': {'weight': 0.005, 'default': 0.0, 'payload_key': 'DiabetesMellitus'},
            'ejection_fraction': {'weight': 0.20, 'default': 55.0, 'payload_key': 'EjectionFraction'},
            'high_blood_pressure': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Hypertension'},
            'platelets': {'weight': 0.03, 'default': 250000.0, 'payload_key': 'Platelets'},
            'serum_creatinine': {'weight': 0.22, 'default': 0.9, 'payload_key': 'SerumCreatinine'},
            'serum_sodium': {'weight': 0.05, 'default': 140.0, 'payload_key': 'SerumSodium'},
            'sex': {'weight': 0.01, 'default': 1.0, 'payload_key': 'Gender'},
            'smoking': {'weight': 0.005, 'default': 0.0, 'payload_key': 'Smoking'},
            'time': {'weight': 0.35, 'default': 200.0, 'payload_key': 'FollowUpTime'}
        }
    },
    'Kidney Disease': {
        'model_key': 'kidney',
        'critical_features': ['SerumCreatinine', 'SpecificGravity'],
        'features': {
            'Age': {'weight': 0.05, 'default': 45.0, 'payload_key': 'Age'},
            'BloodPressure': {'weight': 0.04, 'default': 80.0, 'payload_key': 'BloodPressure'},
            'SpecificGravity': {'weight': 0.20, 'default': 1.020, 'payload_key': 'SpecificGravity'},
            'Albumin': {'weight': 0.15, 'default': 0.0, 'payload_key': 'Albumin'},
            'Sugar': {'weight': 0.02, 'default': 0.0, 'payload_key': 'UrineSugar'},
            'RedBloodCells': {'weight': 0.10, 'default': 0.0, 'payload_key': 'RedBloodCells'},
            'PusCell': {'weight': 0.02, 'default': 0.0, 'payload_key': 'PusCell'},
            'PusCellClumps': {'weight': 0.01, 'default': 0.0, 'payload_key': 'PusCellClumps'},
            'Bacteria': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Bacteria'},
            'BloodGlucoseRandom': {'weight': 0.02, 'default': 100.0, 'payload_key': 'BloodGlucoseRandom'},
            'BloodUrea': {'weight': 0.01, 'default': 30.0, 'payload_key': 'BloodUrea'},
            'SerumCreatinine': {'weight': 0.15, 'default': 0.9, 'payload_key': 'SerumCreatinine'},
            'Sodium': {'weight': 0.01, 'default': 140.0, 'payload_key': 'SerumSodium'},
            'Potassium': {'weight': 0.01, 'default': 4.0, 'payload_key': 'Potassium'},
            'Hemoglobin': {'weight': 0.08, 'default': 14.0, 'payload_key': 'Hemoglobin'},
            'PackedCellVolume': {'weight': 0.06, 'default': 42.0, 'payload_key': 'PackedCellVolume'},
            'WhiteBloodCellCount': {'weight': 0.02, 'default': 7000.0, 'payload_key': 'WhiteBloodCellCount'},
            'RedBloodCellCount': {'weight': 0.02, 'default': 5.0, 'payload_key': 'RedBloodCellCount'},
            'Hypertension': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Hypertension'},
            'DiabetesMellitus': {'weight': 0.01, 'default': 0.0, 'payload_key': 'DiabetesMellitus'},
            'CoronaryArteryDisease': {'weight': 0.01, 'default': 0.0, 'payload_key': 'CoronaryArteryDisease'},
            'Appetite': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Appetite'},
            'PedalEdema': {'weight': 0.01, 'default': 0.0, 'payload_key': 'PedalEdema'},
            'Anemia': {'weight': 0.01, 'default': 0.0, 'payload_key': 'Anaemia'}
        }
    },
    'Heart Disease': {
        'model_key': 'heart',
        'critical_features': ['Thalassemia', 'ChestPainType'],
        'features': {
            'Age': {'weight': 0.04, 'default': 55.0, 'payload_key': 'Age'},
            'Sex': {'weight': 0.01, 'default': 1.0, 'payload_key': 'Gender'},
            'ChestPainType': {'weight': 0.18, 'default': 1.0, 'payload_key': 'ChestPainType'},
            'RestingBP': {'weight': 0.02, 'default': 120.0, 'payload_key': 'BloodPressure'},
            'Cholesterol': {'weight': 0.01, 'default': 200.0, 'payload_key': 'Cholesterol'},
            'FastingBloodSugar': {'weight': 0.01, 'default': 0.0, 'payload_key': 'FastingBloodSugar'},
            'RestECG': {'weight': 0.01, 'default': 0.0, 'payload_key': 'RestECG'},
            'MaxHeartRate': {'weight': 0.12, 'default': 150.0, 'payload_key': 'MaxHeartRate'},
            'ExerciseAngina': {'weight': 0.05, 'default': 0.0, 'payload_key': 'ExerciseAngina'},
            'Oldpeak': {'weight': 0.10, 'default': 0.0, 'payload_key': 'Oldpeak'},
            'Slope': {'weight': 0.02, 'default': 1.0, 'payload_key': 'Slope'},
            'NumMajorVessels': {'weight': 0.20, 'default': 0.0, 'payload_key': 'NumMajorVessels'},
            'Thalassemia': {'weight': 0.23, 'default': 2.0, 'payload_key': 'Thalassemia'}
        }
    },
    'Stroke': {
        'model_key': 'stroke',
        'critical_features': ['age'],
        'features': {
            'gender': {'weight': 0.01, 'default': 1.0, 'payload_key': 'Gender'},
            'age': {'weight': 0.45, 'default': 45.0, 'payload_key': 'Age'},
            'hypertension': {'weight': 0.05, 'default': 0.0, 'payload_key': 'Hypertension'},
            'heart_disease': {'weight': 0.04, 'default': 0.0, 'payload_key': 'CoronaryArteryDisease'},
            'ever_married': {'weight': 0.01, 'default': 1.0, 'payload_key': 'EverMarried'},
            'work_type': {'weight': 0.02, 'default': 2.0, 'payload_key': 'WorkType'},
            'Residence_type': {'weight': 0.01, 'default': 1.0, 'payload_key': 'ResidenceType'},
            'avg_glucose_level': {'weight': 0.25, 'default': 90.0, 'payload_key': 'Glucose'},
            'bmi': {'weight': 0.14, 'default': 22.0, 'payload_key': 'BMI'},
            'smoking_status': {'weight': 0.02, 'default': 0.0, 'payload_key': 'Smoking'}
        }
    },
    'Liver Disease': {
        'model_key': 'liver',
        'positive_class_index': 0, 
        'critical_features': ['Direct_bilirubin', 'Alamine_Aminotransferase'],
        'features': {
            'age': {'weight': 0.05, 'default': 45.0, 'payload_key': 'Age'},
            'gender': {'weight': 0.01, 'default': 1.0, 'payload_key': 'Gender'},
            'Total_bilirubin': {'weight': 0.08, 'default': 0.8, 'payload_key': 'TotalBilirubin'},
            'Direct_bilirubin': {'weight': 0.25, 'default': 0.2, 'payload_key': 'DirectBilirubin'},
            'Alkaline_Phosphotase': {'weight': 0.15, 'default': 80.0, 'payload_key': 'AlkalinePhosphotase'},
            'Alamine_Aminotransferase': {'weight': 0.20, 'default': 25.0, 'payload_key': 'AlamineAminotransferase'},
            'Aspartate_Aminotransferase': {'weight': 0.15, 'default': 25.0, 'payload_key': 'AspartateAminotransferase'},
            'Total_Proteins': {'weight': 0.05, 'default': 7.0, 'payload_key': 'TotalProteins'},
            'Albumin': {'weight': 0.04, 'default': 4.0, 'payload_key': 'Albumin'},
            'Albumin_and_Globulin_Ratio': {'weight': 0.02, 'default': 1.0, 'payload_key': 'AlbuminGlobulinRatio'}
        }
    }
}

def is_missing(val, payload_key):
    """Determines if a value is genuinely missing or a default/impossible 0."""
    if val in [None, '', 'null']:
        return True
    try:
        float_val = float(val)
        # List of features where a '0' input is physically impossible and means the UI box was untouched
        impossible_zeros = [
            'Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness',
            'FollowUpTime', 'SerumCreatinine', 'EjectionFraction', 'SerumSodium',
            'CreatininePhosphokinase', 'Platelets', 'SpecificGravity', 'BloodGlucoseRandom',
            'BloodUrea', 'Potassium', 'Hemoglobin', 'PackedCellVolume', 'WhiteBloodCellCount',
            'RedBloodCellCount', 'Cholesterol', 'MaxHeartRate', 'TotalBilirubin',
            'DirectBilirubin', 'AlkalinePhosphotase', 'AlamineAminotransferase',
            'AspartateAminotransferase', 'TotalProteins', 'Albumin'
        ]
        if float_val == 0.0 and payload_key in impossible_zeros:
            return True
    except ValueError:
        return True
    return False

def generate_elaborate_6month_routine(evaluated_diseases, patient_vitals, alert_biomarkers):
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        return '<p style="color: #d9534f;"><strong>Error:</strong> OpenRouter API key is not configured.</p>'

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
       - <strong>Targeted Nutrition & Micronutrient Constraints</strong>
       - <strong>Prescribed Physical Regimen</strong>
       - <strong>Direct Biomarker Impact</strong>
    3. Include a final section: <strong>Critical Clinical Retesting Schedule</strong>.
    4. NO medical prescription of pharmaceutical drugs. 
    5. Output CLEAN HTML only using <h3>, <h4>, <p>, <ul>, <li> tags. Do not wrap in markdown code fences.
    """

    payload = {
        'model': 'openrouter/free',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.2,
        'max_tokens': 550, # Dropped from 950 to avoid Vercel timeout
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
        # Aggressive 6.5 second timeout to guarantee it finishes before Vercel kills the app
        with urllib.request.urlopen(req, timeout=6.5) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            if 'choices' in res_data and len(res_data['choices']) > 0:
                return res_data['choices'][0]['message']['content']
            return f"<p style='color: #d9534f;'>Generation failed.</p>"
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        return f"<p style='color: #d9534f;'><strong>OpenRouter API Error ({e.code}):</strong> {error_body}</p>"
    except Exception as e:
        return f"<p style='color: #d9534f;'><strong>Routine generation error:</strong> {str(e.__class__.__name__)} - {str(e)}</p>"

@app.route('/api/predict/general', methods=['POST'])
def predict_general():
    if load_errors:
        return jsonify({'status': 'error', 'message': f'Model load error: {load_errors}'}), 500

    data = request.json or {}
    evaluated_diseases = {}
    biomarker_alerts = {}

    for disease_name, meta in DISEASE_METADATA.items():
        model = models.get(meta['model_key'])
        if not model:
            continue

        # 1. Reject if ANY critical feature is missing (or is an impossible 0)
        missing_critical = False
        for crit in meta['critical_features']:
            payload_key = meta['features'][crit]['payload_key']
            val = data.get(payload_key)
            if is_missing(val, payload_key):
                missing_critical = True
                break
                
        if missing_critical:
            continue

        # 2. Build row in exact column order, apply safe defaults, calculate confidence
        feature_row = {}
        present_weight = 0.0

        for feat_name, feat_meta in meta['features'].items():
            payload_key = feat_meta['payload_key']
            val = data.get(payload_key)
            
            if not is_missing(val, payload_key):
                try:
                    parsed_val = float(val)
                    feature_row[feat_name] = parsed_val
                    present_weight += feat_meta['weight']

                    # Out-of-range clinical flags
                    if (
                        (feat_name == 'Glucose' and parsed_val >= 126.0) or
                        (feat_name == 'BMI' and parsed_val >= 30.0) or
                        (feat_name == 'BloodPressure' and parsed_val >= 140.0) or
                        (feat_name == 'SerumCreatinine' and parsed_val >= 1.4) or
                        (feat_name == 'Direct_bilirubin' and parsed_val >= 0.5) or
                        (feat_name == 'ejection_fraction' and parsed_val <= 40.0)
                    ):
                        biomarker_alerts[payload_key] = parsed_val
                except ValueError:
                    feature_row[feat_name] = feat_meta['default']
            else:
                # Apply the safe default and DO NOT increase confidence weight
                feature_row[feat_name] = feat_meta['default']

        confidence_score = round(min(present_weight * 100, 100.0), 1)

        # 3. Reject if overall data quality is too low (less than 65% of required feature importance)
        if confidence_score < 65.0:
            continue

        # 4. Generate Inference
        try:
            df_input = pd.DataFrame([feature_row])
            pos_idx = meta.get('positive_class_index', 1) 
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
            'message': 'Insufficient clinical parameters provided. Please input critical diagnostic markers.'
        })

    ai_routine = generate_elaborate_6month_routine(evaluated_diseases, data, biomarker_alerts)

    return jsonify({
        'status': 'success',
        'predictions': evaluated_diseases,
        'biomarker_alerts': biomarker_alerts,
        'routine': ai_routine,
    })

if __name__ == '__main__':
    app.run(debug=True)
