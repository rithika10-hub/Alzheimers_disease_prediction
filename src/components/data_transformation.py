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
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create a data preprocessing pipeline for numerical and categorical features.
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

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            logger.error(f"Error in get_data_transformer_object: {str(e)}")
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Apply transformations to the dataset and save the preprocessing object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train and test data loaded successfully.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Diagnosis"

            # Drop unwanted columns if they exist
            columns_to_drop = ["PatientID", "DoctorInCharge"]
            train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], inplace=True)
            test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns], inplace=True)

            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise KeyError(f"Target column '{target_column_name}' is missing in dataset.")

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing transformation...")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            logger.info("Preprocessing object saved successfully.")

            # ✅ FIX: Ensure function returns three values
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logger.error(f"Error in initiate_data_transformation: {str(e)}")
            raise customexception(e, sys)
