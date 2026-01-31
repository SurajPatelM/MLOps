from fastapi import FastAPI, status, HTTPException, Query
from pydantic import BaseModel
from predict import available_models, predict_dt, predict_rf, predict_lr, predict_proba

app = FastAPI()

SPECIES = ["setosa", "versicolor", "virginica"]

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    class_id: int
    species: str
    confidence: float
    probs: list
    model: str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.get("/models", status_code=status.HTTP_200_OK)
async def list_models():
    return {"available_models": available_models()}

def _build_features(iris_features: IrisData):
    """
    Convert request body into model input feature array.
    Args:
        iris_features (IrisData): Input iris features.
    Returns:
        features (list): Feature array in sklearn iris order.
    """
    features = [[
        iris_features.sepal_length,
        iris_features.sepal_width,
        iris_features.petal_length,
        iris_features.petal_width
    ]]
    return features

def _format_response(model_key, pred, proba):
    """
    Build response with class id, species name, confidence, and probabilities.
    Args:
        model_key (str): Model identifier ("dt", "rf", "lr").
        pred (numpy.ndarray): Predicted class label array.
        proba (numpy.ndarray): Predicted probability array.
    Returns:
        response (IrisResponse): Formatted response model.
    """
    class_id = int(pred[0])

    if class_id < 0 or class_id >= len(SPECIES):
        raise ValueError(f"Invalid class_id predicted: {class_id}")

    probs = [float(x) for x in proba[0].tolist()]
    confidence = float(max(probs))

    return IrisResponse(
        class_id=class_id,
        species=SPECIES[class_id],
        confidence=confidence,
        probs=probs,
        model=model_key
    )

def _predict_with_model(model_key, iris_features: IrisData):
    """
    Run prediction + probability for a selected model.
    Args:
        model_key (str): Model identifier ("dt", "rf", "lr").
        iris_features (IrisData): Input iris features.
    Returns:
        response (IrisResponse): Prediction response.
    """
    features = _build_features(iris_features)

    if model_key == "dt":
        pred = predict_dt(features)
    elif model_key == "rf":
        pred = predict_rf(features)
    elif model_key == "lr":
        pred = predict_lr(features)
    else:
        raise ValueError(f"Invalid model_key '{model_key}'")

    proba = predict_proba(model_key, features)
    return _format_response(model_key, pred, proba)

@app.post("/predict", response_model=IrisResponse)
async def predict_any(
    iris_features: IrisData,
    model: str = Query("dt", description="Model key: dt, rf, lr")
):
    try:
        if model not in available_models():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model='{model}'. Use one of: {available_models()}"
            )

        return _predict_with_model(model, iris_features)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/dt", response_model=IrisResponse)
async def predict_dt_endpoint(iris_features: IrisData):
    try:
        return _predict_with_model("dt", iris_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/rf", response_model=IrisResponse)
async def predict_rf_endpoint(iris_features: IrisData):
    try:
        return _predict_with_model("rf", iris_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lr", response_model=IrisResponse)
async def predict_lr_endpoint(iris_features: IrisData):
    try:
        return _predict_with_model("lr", iris_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
