# src/utils.py
import pandas as pd
from sklearn.metrics import mean_squared_error

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using mean squared error."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def save_model(model, filepath):
    """Save the trained model to a file."""
    import joblib
    joblib.dump(model, filepath)
