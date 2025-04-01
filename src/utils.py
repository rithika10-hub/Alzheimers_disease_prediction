import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.loggs.logger import logger
from src.exception.exception import customexception

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Accuracy scores
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            report[model_name] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "precision": precision_score(y_test, y_test_pred),
                "recall": recall_score(y_test, y_test_pred),
                "f1_score": f1_score(y_test, y_test_pred)
            }
        
        return report
    except Exception as e:
        logger.info('Exception occurred during model evaluation')
        raise customexception(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.info('Exception occurred in load_object function')
        raise customexception(e, sys)
