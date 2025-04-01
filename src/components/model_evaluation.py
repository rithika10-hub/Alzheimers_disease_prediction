import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.loggs.logger import logger
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
        logger.info("Alzheimer's Disease Model Evaluation Started")
        print("Alzheimer's Disease Model Evaluation Initialized.")

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        logger.info("Evaluation metrics calculated: Accuracy, Precision, Recall, F1-Score")
        print(f"Evaluation Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
        return accuracy, precision, recall, f1

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model_path = os.path.join("artifacts", "alzheimers_model.pkl")
            model = load_object(model_path)

            logger.info("Model loaded for evaluation")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(X_test)
                
                accuracy, precision, recall, f1 = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                logger.info(f"Metrics logged: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}")

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="alzheimers_disease_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logger.error("Exception occurred during model evaluation")
            raise customexception(e, sys)