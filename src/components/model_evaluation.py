import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import load_object
from src.loggs.logger import logger  # Ensure correct import
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
        logger.info("Model evaluation process started.")
        mlflow.set_tracking_uri("   ")  # Set MLflow tracking URI

    def eval_metrics(self, actual, pred):
        """
        Compute evaluation metrics: RMSE, MAE, R² score.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logger.info(f"Evaluation Metrics - RMSE: {rmse}, MAE: {mae}, R²: {r2}")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def initiate_model_evaluation(self, test_array):
        """
        Evaluate the model using test data.
        """
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Load the trained model
            model_path = os.path.join("artifacts", "model.pkl")
            if not os.path.exists(model_path):
                logger.error(f"Model file missing at {model_path}. Please train the model first.")
                return None  # Stop execution gracefully
            
            model = load_object(model_path)
            logger.info("Model loaded successfully.")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logger.info(f"MLflow Tracking URI: {tracking_url_type_store}")

            with mlflow.start_run():
                predictions = model.predict(X_test)
                metrics = self.eval_metrics(y_test, predictions)

                mlflow.log_metric("rmse", metrics["rmse"])
                mlflow.log_metric("r2", metrics["r2"])
                mlflow.log_metric("mae", metrics["mae"])

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="alzheimers_ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                logger.info("Model evaluation completed and logged in MLflow.")

            return metrics  # Return evaluation metrics

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise customexception(e, sys)
