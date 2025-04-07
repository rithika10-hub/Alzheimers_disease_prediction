import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 👇 Replace with your actual dataset path
DATA_PATH = r"D:\datascience projects\ml project\Alzheimers_disease_prediction\notebook\Alzheimers_disease_data.csv"

# Load data
df = pd.read_csv(DATA_PATH)
print("Dataset Columns:", df.columns)

# Split into features and target
X = df.drop(columns=['Diagnosis', 'PatientID', 'DoctorInCharge'])  # Adjust as needed
y = df['Diagnosis']

print(f"✅ Features Shape: {X.shape}")
print(f"✅ Target Shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replace 'XXXConfid' with NaN and fix FutureWarnings
X_train = X_train.replace('XXXConfid', np.nan).infer_objects(copy=False)
X_test = X_test.replace('XXXConfid', np.nan).infer_objects(copy=False)

# Fill missing values
X_train = X_train.ffill()
X_test = X_test.ffill()

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")
