from pathlib import Path
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data import load_data, split_data

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def fit_model(X_train, y_train):
    """
    Train a Decision Tree, Random Forest and Logistic Regression classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    joblib.dump(dt_classifier, MODEL_DIR / "iris_dt.pkl")

    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=12, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier, MODEL_DIR / "iris_rf.pkl")

    lr_classifier = LogisticRegression(max_iter=500, random_state=12)
    lr_classifier.fit(X_train, y_train)
    joblib.dump(lr_classifier, MODEL_DIR / "iris_lr.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)

    for p in sorted(MODEL_DIR.glob("iris_*.pkl")):
        print(" -", p)
