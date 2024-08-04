import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import mlflow.sklearn
from utils import load_data, save_model

print("MLFlow Version = "+mlflow.__version__)

# Load the dataset
# X, y = load_data('data/diabetes.csv')
X, y = load_data('data_git/diabetes.csv')

def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    mse = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return -mse

def perform_hyperparameter_tuning_and_logging():
    mlflow.start_run(run_name="Hyperparameter Tuning Run")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print("Best parameters: ", best_params)

    # Save the best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    save_model(best_model, 'models/best_model.pkl')

    # Log parameters and metrics to MLflow
    mlflow.log_params(best_params)
    mse = cross_val_score(best_model, X_test, y_test, cv=3, scoring='neg_mean_squared_error').mean()
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    for run in range(3):  # Conduct three runs
        perform_hyperparameter_tuning_and_logging()
