import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
import joblib

def objective(trial):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    accuracy = cross_val_score(model, X_train, y_train, cv=3).mean()
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best parameters: ", study.best_params)
    
    # Save the best model
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    best_model = RandomForestClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, 'models/best_model.pkl')
