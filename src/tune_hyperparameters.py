import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from utils import load_data, save_model

# Load the dataset
X, y = load_data('data/diabetes.csv')

def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    mse = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return -mse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best parameters: ", study.best_params)
    
    # Save the best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = RandomForestRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    save_model(best_model, 'models/best_model.pkl')
