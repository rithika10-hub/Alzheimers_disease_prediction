import logging
import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception.exception import customexception
from src.loggs.logger import logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", 
                "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal", 
                "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", 
                "MMSE", "FunctionalAssessment", "ADL"
            ]
            categorical_columns = [
                "Gender", "Ethnicity", "EducationLevel", "Smoking", "FamilyHistoryAlzheimers",
                "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension",
                "MemoryComplaints", "BehavioralProblems", "Confusion", "Disorientation",
                "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"
            ]
            
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
            ])
            
            logger.info("Creating column transformer with preprocessing pipelines")
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            
            return preprocessor
        except Exception as e:
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("Successfully loaded train and test datasets.")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Diagnosis"
            unnecessary_columns = ["PatientID", "DoctorInCharge"]
            
            train_df.drop(columns=[col for col in unnecessary_columns if col in train_df], inplace=True)
            test_df.drop(columns=[col for col in unnecessary_columns if col in test_df], inplace=True)
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logger.info("Applying preprocessing transformations on training and testing data.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
<<<<<<< HEAD
            logger.info("Saving the preprocessing object.")
            save_object(
=======
            logging.info("Saved preprocessing object.")
            
            save_object( # type: ignore
>>>>>>> 411b03acd0e0049701eb568e48a26a212c456793
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return train_arr, test_arr
        except Exception as e:
            raise customexception(e, sys)
