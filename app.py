import streamlit as st
import pandas as pd
import pickle
import os

# Debugging: Check current directory and list files
st.write("### Debug Info")
st.write(f"Current Directory: {os.getcwd()}")
st.write("Files in Directory:", os.listdir())

# Load the trained model (if available)
MODEL_PATH = "artifacts/alzheimers_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

# Streamlit app title
st.title("Alzheimer’s Disease Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your Alzheimer's dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Show basic statistics
    st.write("### Dataset Statistics")
    st.write(df.describe())
    
    # Check if model exists for prediction
    if model is not None:
        st.write("### Make Predictions")
        try:
            # Drop columns that won't be used for prediction
            input_features = df.drop(columns=['Diagnosis', 'DoctorInCharge'], errors='ignore')
            
            # Get the model's expected number of features (assume these are saved during training)
            expected_columns = model.feature_names_in_  # Get the model's feature names
            
            # Check if input data has all required features
            missing_columns = [col for col in expected_columns if col not in input_features.columns]
            extra_columns = [col for col in input_features.columns if col not in expected_columns]

            # If missing columns exist, add them with default values (e.g., 0)
            for col in missing_columns:
                input_features[col] = 0  # You can change this default value
            
            # If there are extra columns, drop them
            input_features = input_features.drop(columns=extra_columns, errors='ignore')
            
            # Ensure the input features are in the same order as expected by the model
            input_features = input_features[expected_columns]

            # Make predictions
            predictions = model.predict(input_features)
            df['Prediction'] = predictions
            
            st.write("### Prediction Results")
            st.dataframe(df[['Prediction']])
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    else:
        st.warning("Model file (alzheimers_model.pkl) not found! Upload a trained model for predictions.")

st.write("Upload a CSV file to analyze the data and make predictions (if a trained model is available).")
