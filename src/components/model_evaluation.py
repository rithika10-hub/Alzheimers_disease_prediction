import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
<<<<<<< HEAD
import pickle
from src.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.loggs.logger import logger
=======
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import load_object
from src.loggs import logging
>>>>>>> 411b03acd0e0049701eb568e48a26a212c456793
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
<<<<<<< HEAD
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
=======
        logging.info("Model evaluation process started.")

    def eval_metrics(self, actual, pred):
        """
        Compute evaluation metrics: RMSE, MAE, R² score.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))  # Root Mean Squared Error
        mae = mean_absolute_error(actual, pred)  # Mean Absolute Error
        r2 = r2_score(actual, pred)  # R² Score
        logging.info(f"Evaluation Metrics - RMSE: {rmse}, MAE: {mae}, R²: {r2}")
        return rmse, mae, r2

    def initiate_model_evaluation(self, test_array):
        """
        Evaluate the model using test data.
        """
        try:
            # Splitting test data into features and target
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Load the trained model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            logging.info("Model loaded successfully.")

            # Get the MLflow tracking URI
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"MLflow Tracking URI: {tracking_url_type_store}")

            # Start MLflow experiment run
            with mlflow.start_run():
                predictions = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)

                # Log metrics to MLflow
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Register model only if not using a file store
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="alzheimers_ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                logging.info("Model evaluation completed and logged in MLflow.")

        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise customexception(e, sys)
>>>>>>> 411b03acd0e0049701eb568e48a26a212c456793
