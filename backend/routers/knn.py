"""KNN router – regression and classification."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from models.schemas import ClassificationRequest
from services.datasets import get_classification_dataset, make_linear_data
from services.ml_engine import train_classification, get_decision_boundary, train_regression
from services.ml_engine import overfit_depth_curve
from services.datasets import get_classification_dataset

router = APIRouter(prefix="/api/knn", tags=["KNN"])


class KNNRequest(BaseModel):
    task: str = "classification"          # classification | regression
    dataset: str = "moons"
    n_neighbors: int = 5
    noise: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    n_samples: int = 300


@router.post("/train")
def train_knn(req: KNNRequest):
    if req.task == "classification":
        X, y = get_classification_dataset(req.dataset, req.noise, req.n_samples, req.random_state)
        res = train_classification(
            X, y,
            model_type="knn",
            n_neighbors=req.n_neighbors,
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
    else:
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        X, y = make_linear_data(n_samples=req.n_samples, noise=req.noise, random_state=req.random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state)
        model = KNeighborsRegressor(n_neighbors=req.n_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        return {
            "metrics": {
                "mse": mse,
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test, y_pred)),
            },
            "scatter": {
                "x_train": X_train.flatten().tolist(),
                "y_train": y_train.tolist(),
                "x_test": X_test.flatten().tolist(),
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
            }
        }


@router.post("/overfit-curve")
def knn_overfit_curve(req: KNNRequest):
    """Return accuracy vs. k curve."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = get_classification_dataset(req.dataset, req.noise, req.n_samples, req.random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    results = []
    for k in range(1, min(31, len(X_train))):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train_s, y_train)
        results.append({
            "k": k,
            "train_score": float(m.score(X_train_s, y_train)),
            "test_score": float(m.score(X_test_s, y_test)),
        })
    return {"curve": results}
