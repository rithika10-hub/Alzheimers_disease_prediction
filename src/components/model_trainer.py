<<<<<<< HEAD
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
=======
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
>>>>>>> 411b03acd0e0049701eb568e48a26a212c456793
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
<<<<<<< HEAD
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
=======
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from src.exception.exception import customexception
from src.loggs import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
>>>>>>> 411b03acd0e0049701eb568e48a26a212c456793
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
<<<<<<< HEAD
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
=======
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoostClassifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier()
            }

            params = {
                "Random Forest": {'n_estimators': [50, 100, 200]},
                "Gradient Boosting": {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [50, 100, 200]},
                "Logistic Regression": {},
                "Decision Tree": {'criterion': ['gini', 'entropy']},
                "XGBClassifier": {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [50, 100, 200]},
                "CatBoostClassifier": {'depth': [6, 8, 10], 'iterations': [50, 100]},
                "AdaBoost Classifier": {'n_estimators': [50, 100, 200]},
                "SVM": {'C': [0.1, 1, 10]},
                "KNN": {'n_neighbors': [3, 5, 7]}
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, param=params)
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise customexception("No sufficiently good model found")
            
            logging.info(f"Best model selected: {best_model_name} with accuracy: {best_model_score:.4f}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy

        except Exception as e:
            raise customexception(e, sys)


>>>>>>> 411b03acd0e0049701eb568e48a26a212c456793
