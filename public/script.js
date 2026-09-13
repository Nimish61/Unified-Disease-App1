async function calculateRisk() {
    // 1. Show loading state on the button (AI takes a few seconds)
    const calcButton = document.querySelector('button'); // Adjust if your button has a specific ID
    const originalText = calcButton.innerText;
    calcButton.innerText = "Analyzing & Generating AI Routine...";
    calcButton.disabled = true;

    // 2. Collect all inputs from the HTML form elements
    const patientData = {
        Age: document.getElementById('age').value,
        Gender: document.getElementById('gender').value, 
        BMI: document.getElementById('bmi').value,
        BloodPressure: document.getElementById('blood_pressure').value,
        Glucose: document.getElementById('glucose').value,
        Insulin: document.getElementById('insulin').value,
        Pregnancies: document.getElementById('pregnancies').value,
        SkinThickness: document.getElementById('skin_thickness').value,
        DiabetesPedigreeFunction: document.getElementById('dpf').value,
        Anaemia: document.getElementById('anaemia').value, 
        CreatininePhosphokinase: document.getElementById('cpk').value,
        DiabetesMellitus: document.getElementById('diabetes_history').value, 
        EjectionFraction: document.getElementById('ejection_fraction').value,
        Hypertension: document.getElementById('hypertension').value, 
        Platelets: document.getElementById('platelets').value,
        SerumCreatinine: document.getElementById('serum_creatinine').value,
        SerumSodium: document.getElementById('serum_sodium').value,
        Smoking: document.getElementById('smoking').value, 
        FollowUpTime: document.getElementById('follow_up_time').value,
        SpecificGravity: document.getElementById('specific_gravity').value,
        Albumin: document.getElementById('albumin').value,
        UrineSugar: document.getElementById('urine_sugar').value,
        RedBloodCells: document.getElementById('rbc').value,
        PusCell: document.getElementById('pus_cell').value,
        PusCellClumps: document.getElementById('pus_cell_clumps').value,
        Bacteria: document.getElementById('bacteria').value,
        BloodGlucoseRandom: document.getElementById('bgr').value,
        BloodUrea: document.getElementById('blood_urea').value,
        Potassium: document.getElementById('potassium').value,
        Hemoglobin: document.getElementById('hemoglobin').value,
        PackedCellVolume: document.getElementById('pcv').value,
        WhiteBloodCellCount: document.getElementById('wbc_count').value,
        RedBloodCellCount: document.getElementById('rbc_count').value,
        CoronaryArteryDisease: document.getElementById('cad_history').value,
        Appetite: document.getElementById('appetite').value,
        PedalEdema: document.getElementById('pedal_edema').value,
        ChestPainType: document.getElementById('chest_pain_type').value,
        Cholesterol: document.getElementById('cholesterol').value,
        FastingBloodSugar: document.getElementById('fbs').value,
        RestECG: document.getElementById('rest_ecg').value,
        MaxHeartRate: document.getElementById('max_heart_rate').value,
        ExerciseAngina: document.getElementById('exercise_angina').value,
        Oldpeak: document.getElementById('oldpeak').value,
        Slope: document.getElementById('slope').value,
        NumMajorVessels: document.getElementById('num_major_vessels').value,
        Thalassemia: document.getElementById('thalassemia').value,
        EverMarried: document.getElementById('ever_married').value,
        WorkType: document.getElementById('work_type').value,
        ResidenceType: document.getElementById('residence_type').value,
        TotalBilirubin: document.getElementById('total_bilirubin').value,
        DirectBilirubin: document.getElementById('direct_bilirubin').value,
        AlkalinePhosphotase: document.getElementById('alkaline_phosphotase').value,
        AlamineAminotransferase: document.getElementById('alt').value,
        AspartateAminotransferase: document.getElementById('ast').value,
        TotalProteins: document.getElementById('total_proteins').value,
        AlbuminGlobulinRatio: document.getElementById('agr').value
    };

    try {
        const response = await fetch('/api/predict/general', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(patientData)
        });

        const data = await response.json();
        
        if (data.status === "success") {
            const resultsList = document.getElementById('results-list');
            resultsList.innerHTML = ''; 
            
            for (const [disease, prob] of Object.entries(data.probabilities)) {
                if (prob !== null) {
                    const isHighRisk = prob > 50; 
                    resultsList.innerHTML += `
                        <div class="result-item ${isHighRisk ? 'high-risk' : ''}" style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eef2f5;">
                            <span>${disease} Risk:</span>
                            <strong>${prob}%</strong>
                        </div>
                    `;
                }
            }
            document.getElementById('results-container').style.display = 'block';

            // Render AI Lifestyle Routine
            if (data.routine) {
                const routineBox = document.getElementById('routine-container');
                const routineContent = document.getElementById('routine-content');
                routineContent.innerHTML = data.routine;
                routineBox.style.display = 'block';
                routineBox.scrollIntoView({ behavior: 'smooth' });
            }
        } else {
            alert("Error: " + data.message);
        }

    } catch (error) {
        // Missing block from your code added here
        console.error("Fetch error:", error);
        alert("A network error occurred. Please check the console.");
    } finally {
        // Reset the button after the API call finishes
        calcButton.innerText = originalText;
        calcButton.disabled = false;
    }
}
