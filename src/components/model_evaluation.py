import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import load_object
from src.loggs import logging
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
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
