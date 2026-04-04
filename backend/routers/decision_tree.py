"""Decision Tree router – regression and classification."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from services.datasets import get_classification_dataset, make_linear_data
from services.ml_engine import train_classification, get_decision_boundary, overfit_depth_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

router = APIRouter(prefix="/api/decision-tree", tags=["Decision Tree"])


class DTRequest(BaseModel):
    task: str = "classification"
    dataset: str = "moons"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    criterion: str = "gini"
    noise: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    n_samples: int = 300


@router.post("/train")
def train_dt(req: DTRequest):
    if req.task == "classification":
        X, y = get_classification_dataset(req.dataset, req.noise, req.n_samples, req.random_state)
        res = train_classification(
            X, y,
            model_type="decision_tree",
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
    else:
        X, y = make_linear_data(n_samples=req.n_samples, noise=req.noise, random_state=req.random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state)
        model = DecisionTreeRegressor(max_depth=req.max_depth, min_samples_split=req.min_samples_split, random_state=req.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        return {
            "metrics": {
                "mse": mse,
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test, y_pred)),
            }
        }


@router.post("/overfit-curve")
def dt_overfit_curve(req: DTRequest):
    if req.task == "classification":
        X, y = get_classification_dataset(req.dataset, req.noise, req.n_samples, req.random_state)
    else:
        X, y = make_linear_data(n_samples=req.n_samples, noise=req.noise, random_state=req.random_state)
    curve = overfit_depth_curve(X, y, task=req.task, max_d=20, test_size=req.test_size, random_state=req.random_state)
    return {"curve": curve}
