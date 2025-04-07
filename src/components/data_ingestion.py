import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.exception.exception import customexception

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
    
    def train_model(self, train_arr, test_arr):
        try:
            # Extract feature names from preprocessor
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
                all_feature_names = preprocessor.feature_names_in_.tolist()

            print(f"train_arr shape: {train_arr.shape}")  # Debugging
            print(f"Feature names count: {len(all_feature_names)}")  # Debugging
            
            if len(all_feature_names) != train_arr.shape[1]:  
                raise ValueError(f"Mismatch: train_arr has {train_arr.shape[1]} columns, but all_feature_names has {len(all_feature_names)} columns")
            
            # Create DataFrame
            df_train = pd.DataFrame(train_arr, columns=all_feature_names)
            df_test = pd.DataFrame(test_arr, columns=all_feature_names)

            # Splitting into X, y
            X_train = df_train.iloc[:, :-1]
            y_train = df_train.iloc[:, -1]
            X_test = df_test.iloc[:, :-1]
            y_test = df_test.iloc[:, -1]

            # Train the model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Save model
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)
            
            return self.model_path
        except Exception as e:
            raise customexception(e, sys)

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            model_path = self.train_model(train_arr, test_arr)
            return model_path
        except Exception as e:
            raise customexception(e, sys)