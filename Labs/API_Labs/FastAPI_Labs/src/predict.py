import joblib
from pathlib import Path
from functools import lru_cache

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"

MODEL_FILES = {
    "dt": "iris_dt.pkl",
    "rf": "iris_rf.pkl",
    "lr": "iris_lr.pkl",
}

def available_models():
    """
    Return the list of available model keys.
    Args:
        None
    Returns:
        models (list): List of available model keys.
    """
    return sorted(MODEL_FILES.keys())

@lru_cache(maxsize=None)
def _load_model(model_key):
    """
    Load and cache the model for the given model key.
    Args:
        model_key (str): Model identifier ("dt", "rf", "lr").
    Returns:
        model (object): Loaded scikit-learn model.
    """
    if model_key not in MODEL_FILES:
        raise ValueError(f"Unknown model_key '{model_key}'. Use one of: {available_models()}")

    model_path = MODEL_DIR / MODEL_FILES[model_key]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Run 'python train.py' from src/ to generate models."
        )

    return joblib.load(model_path)

def predict_dt(X):
    """
    Predict the class labels for the input data using Decision Tree model.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = _load_model("dt")
    y_pred = model.predict(X)
    return y_pred

def predict_rf(X):
    """
    Predict the class labels for the input data using Random Forest model.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = _load_model("rf")
    y_pred = model.predict(X)
    return y_pred

def predict_lr(X):
    """
    Predict the class labels for the input data using Logistic Regression model.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = _load_model("lr")
    y_pred = model.predict(X)
    return y_pred

def predict_proba(model_key, X):
    """
    Predict class probabilities for the input data using a selected model.
    Args:
        model_key (str): Model identifier ("dt", "rf", "lr").
        X (numpy.ndarray): Input data for which probabilities are to be made.
    Returns:
        y_proba (numpy.ndarray): Predicted class probabilities.
    """
    model = _load_model(model_key)
    if not hasattr(model, "predict_proba"):
        raise NotImplementedError(f"Model '{model_key}' does not support predict_proba().")
    y_proba = model.predict_proba(X)
    return y_proba

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    # Backward-compatible default: Decision Tree
    return predict_dt(X)
