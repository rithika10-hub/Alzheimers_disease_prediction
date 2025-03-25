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


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """
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
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
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
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Diagnosis"
            
            # Drop unwanted columns
            train_df = train_df.drop(columns=["PatientID", "DoctorInCharge"], errors='ignore')
            test_df = test_df.drop(columns=["PatientID", "DoctorInCharge"], errors='ignore')
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing data.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object.")
            
            save_object( # type: ignore
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise customexception(e, sys)
