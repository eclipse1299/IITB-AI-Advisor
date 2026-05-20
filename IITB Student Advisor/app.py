import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load env vars manually
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass

if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# load models
try:
    with open(os.path.join(BASE_DIR, "knn_imputer.pkl"), "rb") as f:
        knn_imputer = pickle.load(f)
    with open(os.path.join(BASE_DIR, "input_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    print("Models loaded OK")
except Exception as e:
    print(f"Startup error: {e}")
    knn_imputer = None
    scaler = None
    rf_model = None
    feature_names = []

def get_priorities(student_data):
    stats = {
        'StudyHoursPerDay': {'median': 3.5, 'p75': 4.5, 'dir': 1, 'name': 'Study Hours per Day'},
        'SocialMediaHours': {'median': 2.5, 'p75': 3.3, 'dir': -1, 'name': 'Social Media Hours per Day'},
        'PassiveEntertainmentHrs': {'median': 1.8, 'p75': 2.5, 'dir': -1, 'name': 'Passive Entertainment Hours'},
        'AttendancePercentage': {'median': 84.4, 'p75': 85.0, 'dir': 1, 'name': 'Attendance Percentage'}, # Lowered threshold to 85% to stop AI nagging
        'SleepHoursPerNight': {'median': 6.5, 'p75': 7.3, 'dir': 1, 'name': 'Sleep Hours per Night'},
        'Exercise_frequency': {'median': 3.0, 'p75': 4.0, 'dir': 1, 'name': 'Exercise Frequency'},
        'mental_health_rating': {'median': 6.0, 'p75': 7.0, 'dir': 1, 'name': 'Mental Health Rating'},
        'ProductivityScore': {'median': 4.45, 'p75': 6.82, 'dir': 1, 'name': 'Productivity Score'},
        'TotalScreenTime': {'median': 4.2, 'p75': 5.4, 'dir': -1, 'name': 'Total Screen Time'}
    }
    
    weights = {
        'StudyHoursPerDay': 0.4982,
        'ProductivityScore': 0.2216,
        'mental_health_rating': 0.1165,
        'TotalScreenTime': 0.0257,
        'SleepHoursPerNight': 0.0246,
        'AttendancePercentage': 0.0186,
        'SocialMediaHours': 0.0185,
        'PassiveEntertainmentHrs': 0.0181,
        'Exercise_frequency': 0.0068
    }
    
    issues = []
    for col, info in stats.items():
        val = student_data.get(col)
        if pd.isna(val): continue 
        
        val = float(val)
        w = weights.get(col, 1.0)
        
        if info['dir'] == 1:
            if val < info['p75']:
                score = w * ((info['p75'] - val) / info['p75'])
                issues.append({'col': col, 'name': info['name'], 'score': score, 'val': val, 'bench': info['p75'], 'type': 'low'})
        else:
            if val > info['median']:
                score = w * ((val - info['median']) / info['median'])
                issues.append({'col': col, 'name': info['name'], 'score': score, 'val': val, 'bench': info['median'], 'type': 'high'})
                
    issues.sort(key=lambda x: x['score'], reverse=True)
    return issues

def generate_ai_advice(grade, raw_data):
    risk = "At-Risk" if grade < 55 else ("Average" if grade < 80 else "Excellent")
    
    # 1. Get the bad habits
    issues = get_priorities(raw_data)
    
    # 2. Find the good habits (anything NOT in the issues list!)
    issue_cols = [item['col'] for item in issues]
    strengths = []
    
    if "TotalScreenTime" not in issue_cols and pd.notna(raw_data.get("TotalScreenTime")):
        strengths.append(f"- Highly optimized Screen Time ({raw_data['TotalScreenTime']} hrs)")
    if "SleepHoursPerNight" not in issue_cols and pd.notna(raw_data.get("SleepHoursPerNight")):
        strengths.append(f"- Excellent Sleep Routine ({raw_data['SleepHoursPerNight']} hrs)")
    if "StudyHoursPerDay" not in issue_cols and pd.notna(raw_data.get("StudyHoursPerDay")):
        strengths.append(f"- Strong Academic Focus ({raw_data['StudyHoursPerDay']} hrs)")
    
    # 3. Format the lists for the LLM
    target_issues = []
    for item in issues[:3]:  
        if item['type'] == 'low':
            target_issues.append(f"- {item['name']}: {item['val']:.1f} (Needs to be {item['bench']:.1f}+)")
        else:
            target_issues.append(f"- {item['name']}: {item['val']:.1f} (Should be under {item['bench']:.1f})")
            
    issue_str = "\n".join(target_issues) if target_issues else "None!"
    strength_str = "\n".join(strengths[:2]) if strengths else "None specifically identified." 
    
    sys_prompt = (
        "You are an empathetic academic advisor at IIT Bombay. "
        "RULE 1: Your first sentence must state the student's predicted Cumulative Grade out of 100 and their risk level. "
        "RULE 2: Provide a short 2-3 sentence personalized recommendation using 'you'. "
        "RULE 3: Plain text only. No markdown or bolding. "
        "RULE 4: Address the top deficits. If the student only has 1 or 0 deficits, you MUST balance your advice by praising the specific metrics listed in their Strengths."
        "RULE 5: If a student's deficit is mental health(and is very low) suggest them to go to the Student Wellness Center"
    )
    
    user_prompt = f"Predicted Grade: {grade:.1f}/100 ({risk})\n\nDeficits to Fix:\n{issue_str}\n\nStrengths to Praise:\n{strength_str}\n\nProvide short advice."
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "System Error: Missing API Key. Please contact support."

    models = ['gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-09-2025']
    for m in models:
        try:
            model = genai.GenerativeModel(model_name=m, system_instruction=sys_prompt)
            res = model.generate_content(user_prompt)
            if res.text: return res.text.strip()
        except Exception as e:
            print(f"Model {m} failed: {e}")
            continue 
            
    # Graceful fallback text to satisfy Requirement R5 if all API calls fail
    return "The AI Advisor is currently experiencing high traffic. Please try again in a few moments."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not all([scaler, rf_model, knn_imputer]):
        return jsonify({"error": "Pipeline missing pickle files."}), 500
    
    try:
        req = request.json or {}
        
        def parse_val(key, is_num=True):
            val = req.get(key)
            if val == "" or val is None: return np.nan
            return float(val) if is_num else val

        # handle categorical mappings
        diet_map = {"Poor": 0.0, "Fair": 1.0, "Good": 2.0}
        edu_map = {"High School": 0.0, "Bachelor": 1.0, "Master": 2.0}
        net_map = {"Poor": 0.0, "Average": 1.0, "Good": 2.0}
        
        diet_raw = parse_val("Diet_Quality", False)
        edu_raw = parse_val("parental_education_level", False)
        net_raw = parse_val("internet_quality", False)

        diet = diet_map.get(diet_raw, np.nan) if pd.notna(diet_raw) else np.nan
        edu = edu_map.get(edu_raw, np.nan) if pd.notna(edu_raw) else np.nan
        net = net_map.get(net_raw, np.nan) if pd.notna(net_raw) else np.nan
        
        gender_m = 1.0 if req.get("Gender") == "Male" else (0.0 if req.get("Gender") else np.nan)
        gender_o = 1.0 if req.get("Gender") == "Other" else (0.0 if req.get("Gender") else np.nan)
        por_yes = 1.0 if req.get("PoR") == "Yes" else (0.0 if req.get("PoR") else np.nan)
        extra_yes = 1.0 if req.get("extracurricular_participation") == "Yes" else (0.0 if req.get("extracurricular_participation") else np.nan)
        
        sleep = parse_val("SleepHoursPerNight")
        sleep_debt = 1.0 if pd.notna(sleep) and sleep < 6.0 else (0.0 if pd.notna(sleep) else np.nan)
        
        student_data = {
            "Student_ID": 1000.0,
            "Age": parse_val("Age"),
            "StudyHoursPerDay": parse_val("StudyHoursPerDay"),
            "SocialMediaHours": parse_val("SocialMediaHours"),
            "PassiveEntertainmentHrs": parse_val("PassiveEntertainmentHrs"),
            "AttendancePercentage": parse_val("AttendancePercentage"),
            "SleepHoursPerNight": sleep,
            "Diet_Quality": diet,
            "Exercise_frequency": parse_val("Exercise_frequency"),
            "parental_education_level": edu,
            "internet_quality": net,
            "mental_health_rating": parse_val("mental_health_rating"),
            "ProductivityScore": parse_val("ProductivityScore"),
            "TotalScreenTime": parse_val("TotalScreenTime"),
            "Gender_Male": gender_m,
            "Gender_Other": gender_o,
            "PoR_Yes": por_yes,
            "extracurricular_participation_Yes": extra_yes,
            "SleepDebt_Yes": sleep_debt
        }
        
        df = pd.DataFrame([student_data])[feature_names]
        df_imputed = knn_imputer.transform(df)
        df_scaled = scaler.transform(df_imputed)
        
        pred = float(rf_model.predict(df_scaled)[0])
        pred = max(0.0, min(100.0, pred))
        
        status = "at-risk" if pred < 55.0 else ("average" if pred < 80.0 else "good")
        advice = generate_ai_advice(pred, student_data)
        
        return jsonify({
            "predicted_grade": round(pred, 2),
            "status": status,
            "recommendation": advice
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 