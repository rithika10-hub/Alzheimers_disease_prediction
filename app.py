import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("D:\\datascience projects\\ml project\\Alzheimers_disease_prediction\\notebook\\alzheimers_disease_data.csv")
    df = df.drop(columns=["PatientID", "DoctorInCharge"])
    
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, X.columns

model, scaler, feature_names = load_data()

# Streamlit UI
st.title("🧠 Alzheimer's Disease Prediction App")
st.write("Fill in the patient details to predict Alzheimer's disease risk.")

# Define user-friendly input mappings
gender = st.selectbox("Gender", ["Female", "Male"])
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "College", "Graduate"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.slider("Alcohol Consumption (units/week)", 0.0, 50.0, 10.0)
activity = st.slider("Physical Activity (hours/week)", 0.0, 20.0, 5.0)
diet = st.slider("Diet Quality (scale 0–10)", 0.0, 10.0, 5.0)
sleep = st.slider("Sleep Quality (scale 0–10)", 0.0, 10.0, 5.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
age = st.number_input("Age", min_value=50, max_value=100, value=70)

# Health history
family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
cvd = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
depression = st.selectbox("Depression", ["No", "Yes"])
head_injury = st.selectbox("Head Injury", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])

# Vitals & Cognitive metrics
systolic = st.number_input("Systolic BP", 80, 200, 120)
diastolic = st.number_input("Diastolic BP", 40, 120, 80)
chol_total = st.number_input("Total Cholesterol", 100.0, 300.0, 200.0)
chol_ldl = st.number_input("LDL Cholesterol", 50.0, 200.0, 100.0)
chol_hdl = st.number_input("HDL Cholesterol", 20.0, 100.0, 50.0)
triglycerides = st.number_input("Triglycerides", 50.0, 300.0, 150.0)
mmse = st.slider("MMSE Score (0–30)", 0.0, 30.0, 25.0)
functional = st.slider("Functional Assessment (0–10)", 0.0, 10.0, 5.0)
adl = st.slider("ADL (0–10)", 0.0, 10.0, 5.0)

# Symptoms
memory = st.selectbox("Memory Complaints", ["No", "Yes"])
behavior = st.selectbox("Behavioral Problems", ["No", "Yes"])
confusion = st.selectbox("Confusion", ["No", "Yes"])
disorientation = st.selectbox("Disorientation", ["No", "Yes"])
personality = st.selectbox("Personality Changes", ["No", "Yes"])
tasks = st.selectbox("Difficulty Completing Tasks", ["No", "Yes"])
forgetfulness = st.selectbox("Forgetfulness", ["No", "Yes"])

# Convert categorical inputs
cat_map = {"No": 0, "Yes": 1, "Female": 0, "Male": 1,
           "White": 0, "Black": 1, "Asian": 2, "Hispanic": 3, "Other": 4,
           "None": 0, "Primary": 1, "Secondary": 2, "College": 3, "Graduate": 4}

input_values = [
    age,
    cat_map[gender],
    cat_map[ethnicity],
    cat_map[education],
    bmi,
    cat_map[smoking],
    alcohol,
    activity,
    diet,
    sleep,
    cat_map[family_history],
    cat_map[cvd],
    cat_map[diabetes],
    cat_map[depression],
    cat_map[head_injury],
    cat_map[hypertension],
    systolic,
    diastolic,
    chol_total,
    chol_ldl,
    chol_hdl,
    triglycerides,
    mmse,
    functional,
    cat_map[memory],
    cat_map[behavior],
    adl,
    cat_map[confusion],
    cat_map[disorientation],
    cat_map[personality],
    cat_map[tasks],
    cat_map[forgetfulness]
]

# Predict
if st.button("Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"🧬 The model predicts **Alzheimer's Disease** with a confidence of {prob:.2%}")
    else:
        st.success(f"✅ The model predicts **No Alzheimer's** with a confidence of {prob:.2%}")
