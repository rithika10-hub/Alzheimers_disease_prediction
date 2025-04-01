import os
import sys
from src.exception.exception import customexception
from src.loggs.logger import logger
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Now the paths are being passed correctly.
            logger.info(f"Initiating data transformation with train data at {train_data_path} and test data at {test_data_path}")
            train_arr, test_arr = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            return train_arr, test_arr
        except Exception as e:
            raise customexception(e, sys)

    def run_pipeline(self, train_data_path, test_data_path):
        try:
            # Step 1: Data Transformation
            train_arr, test_arr = self.initiate_data_transformation(train_data_path, test_data_path)
            
            # Step 2: Model Training
            self.model_trainer.initiate_model_training(train_arr, test_arr)
            
        except Exception as e:
            logger.error("Error occurred during the pipeline execution")
            raise customexception(e, sys)

if __name__ == "__main__":
    # Create an instance of the pipeline
    pipeline = TrainPipeline()
    
    # Correct paths to your train and test data
    train_data_path = "D:/datascience projects/ml project/Alzheimers_disease_prediction/data/train_data.csv"
    test_data_path = "D:/datascience projects/ml project/Alzheimers_disease_prediction/data/test_data.csv"
    
    # Run the pipeline
    pipeline.run_pipeline(train_data_path, test_data_path)
