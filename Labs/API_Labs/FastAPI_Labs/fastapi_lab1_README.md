# Lab 1 Completion — FastAPI Iris Prediction API

## Summary

For Lab 1, I extended the starter FastAPI application (which originally served predictions using a single Decision Tree model) to support multiple trained models and multiple prediction endpoints. The API now serves predictions for the Iris dataset using three classifiers and returns a richer response that includes both the predicted class index and the corresponding Iris species name.

---

## Changes Implemented

### 1) Added New Models

In addition to the original Decision Tree model, I added:

- **Random Forest** (`rf`)
- **Logistic Regression** (`lr`)

---

### 2) Updated Training Script (`src/train.py`)

The original `train.py` trained and saved only one model. I updated it to train and save three models:

| Model | Output File |
|---|---|
| Decision Tree | `model/iris_dt.pkl` |
| Random Forest | `model/iris_rf.pkl` |
| Logistic Regression | `model/iris_lr.pkl` |

After training, the script prints the saved model paths for verification.

---

### 3) Updated Prediction Script (`src/predict.py`)

The original `predict.py` supported a single function (`predict_data`) that loaded one model file and returned a predicted class. I updated it to:

- Add separate prediction functions: `predict_dt(X)`, `predict_rf(X)`, `predict_lr(X)`
- Add `predict_proba(model_key, X)` to return class probabilities (used for confidence and probability outputs)
- Add `available_models()` to list supported model keys (`dt`, `rf`, `lr`)
- Keep `predict_data(X)` for backward compatibility (defaults to Decision Tree prediction)
- Cache loaded models to avoid reloading from disk on every request

---

### 4) Updated API Script (`src/main.py`)

The original `main.py` exposed:

- `GET /` — health check
- `POST /predict` — Decision Tree prediction

I updated the API to include the following endpoints:

#### Model Listing Endpoint

- `GET /models` → returns available model keys (`dt`, `rf`, `lr`)

#### Model-Specific Prediction Endpoints

- `POST /predict/dt` → Decision Tree
- `POST /predict/rf` → Random Forest
- `POST /predict/lr` → Logistic Regression

#### Unified Prediction Endpoint

- `POST /predict?model=dt|rf|lr` — allows selecting the model via a query parameter

---

### 5) Improved Response Schema

Previously, the API response returned only an integer label. Prediction responses now include:

| Field | Description |
|---|---|
| `class_id` | Predicted class index |
| `species` | Predicted class name (`setosa`, `versicolor`, `virginica`) |
| `confidence` | Highest probability among classes |
| `probs` | Probabilities for all three classes |
| `model` | Model key used for the prediction |

---

## Project Structure

```
mlops_labs
└── fastapi_lab1
    ├── assets/
    ├── model/
    │   ├── iris_dt.pkl
    │   ├── iris_rf.pkl
    │   └── iris_lr.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── main.py
    │   ├── predict.py
    │   └── train.py
    ├── Lab1_README.md
    ├── README.md
    └── requirements.txt
```

---

## How to Run

### Step 1: Install Dependencies

From the lab root:

```bash
pip install -r requirements.txt
```

### Step 2: Train Models

Run training from the `src/` folder:

```bash
cd src
python train.py
```

This generates:

- `model/iris_dt.pkl`
- `model/iris_rf.pkl`
- `model/iris_lr.pkl`

### Step 3: Start the FastAPI Server

Run the server from the `src/` folder:

```bash
uvicorn main:app --reload
```

---

## Testing the API

### Swagger UI

Open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Sample Request Body

```json
{
  "petal_length": 1.4,
  "sepal_length": 5.1,
  "petal_width": 0.2,
  "sepal_width": 3.5
}
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/models` | List available models |
| `POST` | `/predict?model=dt\|rf\|lr` | Unified prediction (select model via query param) |
| `POST` | `/predict/dt` | Decision Tree prediction |
| `POST` | `/predict/rf` | Random Forest prediction |
| `POST` | `/predict/lr` | Logistic Regression prediction |

### Example Response

```json
{
  "class_id": 0,
  "species": "setosa",
  "confidence": 0.99,
  "probs": [0.99, 0.01, 0.0],
  "model": "rf"
}
```

---
