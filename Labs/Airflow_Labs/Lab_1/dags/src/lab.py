import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
from kneed import KneeLocator
import pickle
import os
import base64

def _sample_X(X, max_rows=3000, seed=42):
    """
    Return a random subset of X (numpy array) to reduce memory/time.
    """
    n = len(X)
    if n <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx]

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # NOTE: This saves the last-fitted model (k=49), matching your original intent.
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred
    
def train_kmeans_best(data_b64: str, model_filename: str = "kmeans.sav"):
    data_bytes = base64.b64decode(data_b64)
    X = pickle.loads(data_bytes)

    Xs = _sample_X(X, max_rows=3000)  # <<< ADD THIS

    best = {"k": None, "silhouette": -1.0}
    best_model = None

    for k in range(2, 11):
        model = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
        labels = model.fit_predict(Xs)                 # <<< CHANGED
        score = silhouette_score(Xs, labels)           # <<< CHANGED

        if score > best["silhouette"]:
            best = {"k": k, "silhouette": float(score)}
            best_model = model

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    return {"model": model_filename, "best_k": best["k"], "silhouette": best["silhouette"]}


def train_gmm_best(data_b64: str, model_filename: str = "gmm.sav"):
    data_bytes = base64.b64decode(data_b64)
    X = pickle.loads(data_bytes)

    Xs = _sample_X(X, max_rows=3000)  # <<< ADD THIS

    best = {"k": None, "silhouette": -1.0}
    best_model = None

    for k in range(2, 11):
        model = GaussianMixture(n_components=k, random_state=42)
        model.fit(Xs)                         # <<< CHANGED
        labels = model.predict(Xs)            # <<< CHANGED

        score = silhouette_score(Xs, labels)  # <<< CHANGED
        if score > best["silhouette"]:
            best = {"k": k, "silhouette": float(score)}
            best_model = model

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    return {"model": model_filename, "best_k": best["k"], "silhouette": best["silhouette"]}

def write_predictions_csv(data_b64: str, kmeans_info: dict, gmm_info: dict, out_csv: str = "cluster_predictions.csv"):
    data_bytes = base64.b64decode(data_b64)
    X = pickle.loads(data_bytes)
    X = _sample_X(X, max_rows=5000)

    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    kmeans = pickle.load(open(os.path.join(model_dir, kmeans_info["model"]), "rb"))
    gmm = pickle.load(open(os.path.join(model_dir, gmm_info["model"]), "rb"))

    kmeans_labels = kmeans.predict(X)
    gmm_labels = gmm.predict(X)

    df_out = pd.DataFrame({
        "row_id": list(range(len(X))),
        "kmeans_cluster": kmeans_labels,
        "gmm_cluster": gmm_labels
    })

    out_path = os.path.join("/opt/airflow/working_data", out_csv)
    df_out.to_csv(out_path, index=False)

    return {
        "output_csv": out_csv,
        "kmeans_best_k": kmeans_info["best_k"],
        "kmeans_silhouette": kmeans_info["silhouette"],
        "gmm_best_k": gmm_info["best_k"],
        "gmm_silhouette": gmm_info["silhouette"],
        "saved_to": out_path
    }

