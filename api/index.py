from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Determine the absolute path to the models folder in Vercel's environment
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

# Load models (ensure filenames match your actual uploaded files)
def load_model(filename):
    try:
        return pickle.load(open(os.path.join(models_dir, filename), 'rb'))
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

models = {
    'diabetes': load_model('diabetes_model.pkl'),
    'heart_failure': load_model('heart_failure_model.pkl'),
    'kidney': load_model('kidney_model.pkl'),
    'heart': load_model('heart_model.pkl'),
    'stroke': load_model('stroke_model.pkl'),
    'liver': load_model('liver_model.pkl')
}

@app.route('/api/predict/general', methods=['POST'])
def predict_general():
    data = request.json
    results = {}

    # Helper function to safely extract and convert data
    def get_val(key, default=0.0):
        try:
            val = data.get(key)
            return float(val) if val not in [None, ""] else float(default)
        except ValueError:
            return float(default)

    try:
        # 1. DIABETES MAPPING
        if models['diabetes']:
            diabetes_features = np.array([[
                get_val('Pregnancies'), get_val('Glucose'), get_val('BloodPressure'),
                get_val('SkinThickness'), get_val('Insulin'), get_val('BMI'),
                get_val('DiabetesPedigreeFunction'), get_val('Age')
            ]])
            results['Diabetes'] = round(models['diabetes'].predict_proba(diabetes_features)[0][1] * 100, 2)

        # 2. HEART FAILURE MAPPING
        if models['heart_failure']:
            hf_features = np.array([[
                get_val('Age'), get_val('Anaemia'), get_val('CreatininePhosphokinase'),
                get_val('DiabetesMellitus'), get_val('EjectionFraction'), get_val('Hypertension'),
                get_val('Platelets'), get_val('SerumCreatinine'), get_val('SerumSodium'),
                get_val('Gender'), get_val('Smoking'), get_val('FollowUpTime')
            ]])
            results['HeartFailure'] = round(models['heart_failure'].predict_proba(hf_features)[0][1] * 100, 2)

        # 3. KIDNEY MAPPING
        if models['kidney']:
            kidney_features = np.array([[
                get_val('Age'), get_val('BloodPressure'), get_val('SpecificGravity'),
                get_val('Albumin'), get_val('UrineSugar'), get_val('RedBloodCells'),
                get_val('PusCell'), get_val('PusCellClumps'), get_val('Bacteria'),
                get_val('BloodGlucoseRandom'), get_val('BloodUrea'), get_val('SerumCreatinine'),
                get_val('SerumSodium'), get_val('Potassium'), get_val('Hemoglobin'),
                get_val('PackedCellVolume'), get_val('WhiteBloodCellCount'), get_val('RedBloodCellCount'),
                get_val('Hypertension'), get_val('DiabetesMellitus'), get_val('CoronaryArteryDisease'),
                get_val('Appetite'), get_val('PedalEdema'), get_val('Anaemia')
            ]])
            results['KidneyDisease'] = round(models['kidney'].predict_proba(kidney_features)[0][1] * 100, 2)

        # 4. HEART DISEASE MAPPING
        if models['heart']:
            heart_features = np.array([[
                get_val('Age'), get_val('Gender'), get_val('ChestPainType'),
                get_val('BloodPressure'), get_val('Cholesterol'), get_val('FastingBloodSugar'),
                get_val('RestECG'), get_val('MaxHeartRate'), get_val('ExerciseAngina'),
                get_val('Oldpeak'), get_val('Slope'), get_val('NumMajorVessels'), get_val('Thalassemia')
            ]])
            results['HeartDisease'] = round(models['heart'].predict_proba(heart_features)[0][1] * 100, 2)

        # 5. STROKE MAPPING
        if models['stroke']:
            stroke_features = np.array([[
                get_val('Gender'), get_val('Age'), get_val('Hypertension'),
                get_val('CoronaryArteryDisease'), get_val('EverMarried'), get_val('WorkType'),
                get_val('ResidenceType'), get_val('Glucose'), get_val('BMI'), get_val('Smoking')
            ]])
            results['Stroke'] = round(models['stroke'].predict_proba(stroke_features)[0][1] * 100, 2)

        # 6. LIVER MAPPING
        if models['liver']:
            liver_features = np.array([[
                get_val('Age'), get_val('Gender'), get_val('TotalBilirubin'),
                get_val('DirectBilirubin'), get_val('AlkalinePhosphotase'), get_val('AlamineAminotransferase'),
                get_val('AspartateAminotransferase'), get_val('TotalProteins'), get_val('Albumin'),
                get_val('AlbuminGlobulinRatio')
            ]])
            results['LiverDisease'] = round(models['liver'].predict_proba(liver_features)[0][1] * 100, 2)

        return jsonify({"status": "success", "probabilities": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Required by Vercel to expose the Flask app
if __name__ == '__main__':
    app.run(debug=True)