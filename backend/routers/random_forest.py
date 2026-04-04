"""Random Forest router – regression (housing), classification, feature importance, CV."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from services.datasets import get_housing_dataframe, HOUSING_FEATURES, get_classification_dataset
from services.ml_engine import train_classification, get_decision_boundary
from services.cross_validation import run_kfold_cv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

router = APIRouter(prefix="/api/random-forest", tags=["Random Forest"])


class RFRegressionRequest(BaseModel):
    active_features: List[str] = HOUSING_FEATURES
    n_estimators: int = 100
    max_depth: Optional[int] = None
    test_size: float = 0.2
    random_state: int = 42


class RFClassificationRequest(BaseModel):
    dataset: str = "moons"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    noise: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    n_samples: int = 300


class CVRequest(BaseModel):
    active_features: List[str] = HOUSING_FEATURES
    n_splits: int = 5
    n_estimators: int = 100
    random_state: int = 42
    task: str = "regression"
    dataset: str = "housing"


@router.get("/housing-dataset")
def get_housing():
    df = get_housing_dataframe()
    return {
        "features": HOUSING_FEATURES,
        "target": "MedHouseVal",
        "shape": list(df.shape),
        "preview": df.head(5).to_dict(orient="records"),
    }


@router.post("/regression")
def train_rf_regression(req: RFRegressionRequest):
    df = get_housing_dataframe()
    # Validate features
    valid_features = [f for f in req.active_features if f in HOUSING_FEATURES]
    if not valid_features:
        valid_features = HOUSING_FEATURES

    X = df[valid_features].values
    y = df["MedHouseVal"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=req.random_state
    )
    model = RandomForestRegressor(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        random_state=req.random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))

    importances = [
        {"feature": f, "importance": float(imp)}
        for f, imp in sorted(zip(valid_features, model.feature_importances_),
                              key=lambda x: x[1], reverse=True)
    ]

    return {
        "metrics": {
            "mse": mse,
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, y_pred)),
        },
        "feature_importances": importances,
        "active_features": valid_features,
    }


@router.post("/classification")
def train_rf_classification(req: RFClassificationRequest):
    X, y = get_classification_dataset(req.dataset, req.noise, req.n_samples, req.random_state)
    res = train_classification(
        X, y,
        model_type="random_forest",
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        test_size=req.test_size,
        random_state=req.random_state,
    )
    boundary = get_decision_boundary(
        res["model"], res["scaler"],
        res["X_train"], res["X_test"],
        res["y_train"], res["y_test"],
    )
    return {
        "metrics": {k: res[k] for k in ("accuracy", "precision", "recall", "f1", "confusion_matrix")},
        "boundary": boundary,
    }


@router.post("/cross-validate")
def cross_validate(req: CVRequest):
    if req.task == "regression":
        df = get_housing_dataframe()
        valid_features = [f for f in req.active_features if f in HOUSING_FEATURES] or HOUSING_FEATURES
        X = df[valid_features].values
        y = df["MedHouseVal"].values
    else:
        X, y = get_classification_dataset("moons", 0.2, 300, req.random_state)

    results = run_kfold_cv(
        X, y,
        model_type="random_forest",
        task=req.task,
        n_splits=req.n_splits,
        n_estimators=req.n_estimators,
        random_state=req.random_state,
    )
    return {"folds": results, "n_splits": req.n_splits}
