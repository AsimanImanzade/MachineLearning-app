"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel
from typing import List, Optional, Literal


# ─── Generic ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_type: str = "standard"          # standard | lasso | ridge | elasticnet
    alpha: float = 1.0
    l1_ratio: float = 0.5
    noise_level: float = 0.0
    degree: int = 1                        # polynomial degree (overfit sandbox)
    n_samples: int = 200
    test_size: float = 0.2
    random_state: int = 42


class ClassificationRequest(BaseModel):
    dataset: str = "iris"                  # iris | moons | circles | blobs
    n_neighbors: int = 5
    max_depth: Optional[int] = None
    n_estimators: int = 100
    test_size: float = 0.2
    noise: float = 0.2
    random_state: int = 42
    C: float = 1.0
    penalty: str = "l2"


class FeatureImportanceRequest(BaseModel):
    active_features: List[str]
    n_estimators: int = 100
    random_state: int = 42


class CrossValidateRequest(BaseModel):
    model_type: str = "random_forest"
    n_splits: int = 5
    dataset: str = "housing"
    n_estimators: int = 100
    random_state: int = 42


# ─── Responses ────────────────────────────────────────────────────────────────

class RegressionMetrics(BaseModel):
    mse: float
    mae: float
    rmse: float
    r2: float


class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]


class ScatterPoint(BaseModel):
    x: float
    y: float
    predicted: Optional[float] = None
    split: str = "train"          # train | test


class BoundaryGrid(BaseModel):
    xx: List[List[float]]
    yy: List[List[float]]
    zz: List[List[float]]
    x_train: List[float]
    y_train: List[float]
    labels_train: List[int]
    x_test: List[float]
    y_test: List[float]
    labels_test: List[int]


class OverfitSandboxPoint(BaseModel):
    degree: int
    train_mse: float
    test_mse: float


class OverfitDepthPoint(BaseModel):
    depth: int
    train_score: float
    test_score: float


class CVFoldResult(BaseModel):
    fold: int
    train_score: float
    val_score: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class CoefficientItem(BaseModel):
    feature: str
    value: float
