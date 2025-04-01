import os
import sys
import numpy as np
import pandas as pd
from src.exception.exception import customexception
from src.loggs.logger import logger
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from sklearn.ensemble import RandomForestClassifier
import mlflow
from urllib.parse import urlparse

class PredictPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()
        self.model_path = os.path.join("artifacts", "alzheimers_model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    
    def load_model(self):
        try:
            # Load the trained model
            model = load_object(self.model_path)
            logger.info("Model loaded successfully for prediction.")
            return model
        except Exception as e:
            logger.error("Error while loading model.")
            raise customexception(e, sys)
    
    def load_preprocessor(self):
        try:
            # Load the preprocessor object for transforming data
            preprocessor = load_object(self.preprocessor_path)
            logger.info("Preprocessor loaded successfully.")
            return preprocessor
        except Exception as e:
            logger.error("Error while loading preprocessor.")
            raise customexception(e, sys)
    
    def predict(self, input_data: pd.DataFrame):
        try:
            # Load preprocessor and model
            model = self.load_model()
            preprocessor = self.load_preprocessor()

            # Preprocess input data
            logger.info("Preprocessing the input data for prediction.")
            input_features = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(input_features)
            
            # Return the prediction result
            logger.info("Prediction completed.")
            return prediction
        except Exception as e:
            logger.error("Error occurred during prediction.")
            raise customexception(e, sys)

if __name__ == "__main__":
    # Example: Using the pipeline to make predictions
    pipeline = PredictPipeline()

    # Assuming input_data is a pandas DataFrame (you can replace it with actual data)
    input_data = pd.DataFrame({
        'Age': [70],
        'BMI': [25.5],
        'AlcoholConsumption': [2],
        'PhysicalActivity': [1],
        'DietQuality': [3],
        'SleepQuality': [4],
        'SystolicBP': [130],
        'DiastolicBP': [85],
        'CholesterolTotal': [200],
        'CholesterolLDL': [100],
        'CholesterolHDL': [50],
        'CholesterolTriglycerides': [150],
        'MMSE': [28],
        'FunctionalAssessment': [3],
        'ADL': [5],
        'Gender': ['Female'],
        'Ethnicity': ['Caucasian'],
        'EducationLevel': ['High School'],
        'Smoking': ['No'],
        'FamilyHistoryAlzheimers': ['No'],
        'CardiovascularDisease': ['No'],
        'Diabetes': ['No'],
        'Depression': ['No'],
        'HeadInjury': ['No'],
        'Hypertension': ['Yes'],
        'MemoryComplaints': ['Yes'],
        'BehavioralProblems': ['No'],
        'Confusion': ['No'],
        'Disorientation': ['No'],
        'PersonalityChanges': ['Yes'],
        'DifficultyCompletingTasks': ['Yes'],
        'Forgetfulness': ['Yes']
    })

    # Call predict method
    prediction = pipeline.predict(input_data)
    print("Prediction Result:", prediction)
