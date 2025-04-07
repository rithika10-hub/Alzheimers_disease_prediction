import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
    """Save a Python object to a file using pickle."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)

def load_object(file_path):
    """Load a Python object from a pickle file."""
    with open(file_path, "rb") as file:
        return pickle.load(file)

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    """Train and evaluate multiple models, returning their performance metrics."""
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        }
    
    return results

def preprocess_data(df, target_column=None):
    """Preprocess dataset by handling missing values and encoding categorical features."""
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing numerical values
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        return X, y
    
    return df