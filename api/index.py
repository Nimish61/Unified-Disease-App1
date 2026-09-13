from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import json
import urllib.request

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
    'liver': load_model('Liver_best_model.joblib')
}

def generate_ai_routine(probabilities, patient_vitals):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "OpenRouter API key is not configured in Vercel environment variables."

    # Identify diseases with higher probabilities
    high_risks = {k: v for k, v in probabilities.items() if v >= 20.0}
    
    prompt = f"""
    You are an AI clinical wellness and lifestyle advisor. Based on the machine learning risk assessment and vitals below, generate a clear, actionable, and encouraging health routine.

    PATIENT VITALS:
    - Age: {patient_vitals.get('Age', 'N/A')}
    - BMI: {patient_vitals.get('BMI', 'N/A')}
    - Blood Pressure: {patient_vitals.get('BloodPressure', 'N/A')} mmHg
    - Glucose: {patient_vitals.get('Glucose', 'N/A')} mg/dL
    - Smoking: {'Yes' if patient_vitals.get('Smoking') == 1 else 'No'}

    ASSESSED DISEASE RISKS:
    {json.dumps(probabilities, indent=2)}

    HIGH RISK ALERTS (>= 20%):
    {json.dumps(high_risks, indent=2) if high_risks else 'All scores under 20% (Low immediate risk)'}

    FORMAT YOUR RESPONSE IN CLEAN HTML (use <h4>, <p>, <ul>, <li>, <strong> tags; do NOT include ```html markdown tags):
    1. Overall Clinical Impression (2-3 sentences)
    2. Nutrition & Dietary Recommendations (focus on identified risks)
    3. Physical Activity & Safe Exercise Plan
    4. Lifestyle Modifications & Routine Habits
    5. Follow-up Diagnostic Tests to Discuss with a Doctor
    """

    payload = {
        "model": "google/gemini-2.5-flash", # You can also use "openai/gpt-4o-mini"
        "messages": [
            {"role": "system", "content": "You are a professional medical lifestyle counselor. Provide preventive lifestyle routines without diagnosing directly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }

    req = urllib.request.Request(
        "[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "[https://unified-disease-app1.vercel.app](https://unified-disease-app1.vercel.app)",
            "X-Title": "VitaPredict Health AI"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<p class='text-danger'>Could not generate routine at this time: {str(e)}</p>"

@app.route('/api/predict/general', methods=['POST'])
def predict_general():
    if load_errors:
        return jsonify({"status": "error", "message": f"Failed to load models: {load_errors}"}), 400

    data = request.json
    results = {}

    def get_val(key, default=0.0):
        try:
            val = data.get(key)
            return float(val) if val not in [None, ""] else float(default)
        except ValueError:
            return float(default)

    try:
        if models['diabetes']:
            diabetes_df = pd.DataFrame([{
                'Pregnancies': get_val('Pregnancies'), 'Glucose': get_val('Glucose'),
                'BloodPressure': get_val('BloodPressure'), 'SkinThickness': get_val('SkinThickness'),
                'Insulin': get_val('Insulin'), 'BMI': get_val('BMI'),
                'DiabetesPedigreeFunction': get_val('DiabetesPedigreeFunction'), 'Age': get_val('Age')
            }])
            results['Diabetes'] = round(models['diabetes'].predict_proba(diabetes_df)[0][1] * 100, 2)

        if models['heart_failure']:
            hf_df = pd.DataFrame([{
                'age': get_val('Age'), 'anaemia': get_val('Anaemia'),
                'creatinine_phosphokinase': get_val('CreatininePhosphokinase'),
                'diabetes': get_val('DiabetesMellitus'), 'ejection_fraction': get_val('EjectionFraction'),
                'high_blood_pressure': get_val('Hypertension'), 'platelets': get_val('Platelets'),
                'serum_creatinine': get_val('SerumCreatinine'), 'serum_sodium': get_val('SerumSodium'),
                'sex': get_val('Gender'), 'smoking': get_val('Smoking'), 'time': get_val('FollowUpTime')
            }])
            results['Heart Failure'] = round(models['heart_failure'].predict_proba(hf_df)[0][1] * 100, 2)

        if models['kidney']:
            kidney_df = pd.DataFrame([{
                'Age': get_val('Age'), 'BloodPressure': get_val('BloodPressure'), 'SpecificGravity': get_val('SpecificGravity', 1.0),
                'Albumin': get_val('Albumin'), 'Sugar': get_val('UrineSugar'), 'RedBloodCells': get_val('RedBloodCells'),
                'PusCell': get_val('PusCell'), 'PusCellClumps': get_val('PusCellClumps'), 'Bacteria': get_val('Bacteria'),
                'BloodGlucoseRandom': get_val('BloodGlucoseRandom'), 'BloodUrea': get_val('BloodUrea'), 'SerumCreatinine': get_val('SerumCreatinine'),
                'Sodium': get_val('SerumSodium'), 'Potassium': get_val('Potassium'), 'Hemoglobin': get_val('Hemoglobin'),
                'PackedCellVolume': get_val('PackedCellVolume'), 'WhiteBloodCellCount': get_val('WhiteBloodCellCount'), 'RedBloodCellCount': get_val('RedBloodCellCount'),
                'Hypertension': get_val('Hypertension'), 'DiabetesMellitus': get_val('DiabetesMellitus'), 'CoronaryArteryDisease': get_val('CoronaryArteryDisease'),
                'Appetite': get_val('Appetite'), 'PedalEdema': get_val('PedalEdema'), 'Anemia': get_val('Anaemia')
            }])
            results['Kidney Disease'] = round(models['kidney'].predict_proba(kidney_df)[0][1] * 100, 2)

        if models['heart']:
            heart_df = pd.DataFrame([{
                'Age': get_val('Age'), 'Sex': get_val('Gender'), 'ChestPainType': get_val('ChestPainType'),
                'RestingBP': get_val('BloodPressure'), 'Cholesterol': get_val('Cholesterol'), 'FastingBloodSugar': get_val('FastingBloodSugar'),
                'RestECG': get_val('RestECG'), 'MaxHeartRate': get_val('MaxHeartRate'), 'ExerciseAngina': get_val('ExerciseAngina'),
                'Oldpeak': get_val('Oldpeak'), 'Slope': get_val('Slope'), 'NumMajorVessels': get_val('NumMajorVessels'), 'Thalassemia': get_val('Thalassemia')
            }])
            results['Heart Disease'] = round(models['heart'].predict_proba(heart_df)[0][1] * 100, 2)

        if models['stroke']:
            stroke_df = pd.DataFrame([{
                'gender': get_val('Gender'), 'age': get_val('Age'), 'hypertension': get_val('Hypertension'),
                'heart_disease': get_val('CoronaryArteryDisease'), 'ever_married': get_val('EverMarried'), 'work_type': get_val('WorkType'),
                'Residence_type': get_val('ResidenceType'), 'avg_glucose_level': get_val('Glucose'), 'bmi': get_val('BMI'), 'smoking_status': get_val('Smoking')
            }])
            results['Stroke'] = round(models['stroke'].predict_proba(stroke_df)[0][1] * 100, 2)

        if models['liver']:
            liver_df = pd.DataFrame([{
                'age': get_val('Age'), 'gender': get_val('Gender'), 'Total_bilirubin': get_val('TotalBilirubin'),
                'Direct_bilirubin': get_val('DirectBilirubin'), 'Alkaline_Phosphotase': get_val('AlkalinePhosphotase'), 'Alamine_Aminotransferase': get_val('AlamineAminotransferase'),
                'Aspartate_Aminotransferase': get_val('AspartateAminotransferase'), 'Total_Proteins': get_val('TotalProteins'), 'Albumin': get_val('Albumin'),
                'Albumin_and_Globulin_Ratio': get_val('AlbuminGlobulinRatio')
            }])
            results['Liver Disease'] = round(models['liver'].predict_proba(liver_df)[0][1] * 100, 2)

        # Generate lifestyle routine via OpenRouter
        ai_routine = generate_ai_routine(results, data)

        return jsonify({
            "status": "success",
            "probabilities": results,
            "routine": ai_routine
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
