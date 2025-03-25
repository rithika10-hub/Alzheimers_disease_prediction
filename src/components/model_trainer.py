import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
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


