import pandas as pd
import numpy as np
from src.loggs.logger import logger
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils import save_object, evaluate_model

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'alzheimers_model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logger.info('Splitting dependent and independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTree': DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier(),
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'NaiveBayes': GaussianNB(),
                'GradientBoosting': GradientBoostingClassifier(),
                'XGBoost': XGBClassifier(eval_metric='logloss')
            }
            
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logger.info(f'Model Report: {model_report}')

            # Get the best model based on accuracy
            best_model_name = max(model_report, key=lambda k: model_report[k]['test_accuracy'])
            best_model_score = model_report[best_model_name]['test_accuracy']
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name}, Accuracy Score: {best_model_score}')
            print('\n====================================================================================\n')
            logger.info(f'Best Model Found: {best_model_name}, Accuracy Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logger.info('Exception occurred during model training')
            raise customexception(e, sys)
