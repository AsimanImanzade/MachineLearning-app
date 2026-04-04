"""Core ML training and evaluation engine."""
import numpy as np
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


# ── Regression ────────────────────────────────────────────────────────────────

def build_regression_model(model_type: str, alpha: float = 1.0, l1_ratio: float = 0.5):
    """Factory: return a sklearn regression estimator."""
    mapping = {
        "standard": LinearRegression(),
        "lasso": Lasso(alpha=alpha, max_iter=10000),
        "ridge": Ridge(alpha=alpha),
        "elasticnet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000),
    }
    return mapping.get(model_type, LinearRegression())


def train_regression(X, y, model_type: str, alpha: float, l1_ratio: float,
                     degree: int = 1, test_size: float = 0.2, random_state: int = 42):
    """Train and evaluate a regression model, return metrics + predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = build_regression_model(model_type, alpha, l1_ratio)

    if degree > 1:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_p = poly.fit_transform(X_train)
        X_test_p = poly.transform(X_test)
    else:
        X_train_p, X_test_p = X_train, X_test

    model.fit(X_train_p, y_train)
    y_pred_train = model.predict(X_train_p)
    y_pred_test = model.predict(X_test_p)

    def metrics(y_true, y_pred):
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse": mse,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    coefs = []
    if hasattr(model, "coef_"):
        coef_arr = np.atleast_1d(model.coef_)
        for i, c in enumerate(coef_arr):
            coefs.append({"feature": f"x{i+1}", "value": float(c)})

    return {
        "train_metrics": metrics(y_train, y_pred_train),
        "test_metrics": metrics(y_test, y_pred_test),
        "X_train": X_train.flatten().tolist() if X_train.ndim > 1 else X_train.tolist(),
        "y_train": y_train.tolist(),
        "y_pred_train": y_pred_train.tolist(),
        "X_test": X_test.flatten().tolist() if X_test.ndim > 1 else X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "coefficients": coefs,
    }


def overfit_regression_curve(X_1d, y, max_degree: int = 12, test_size: float = 0.2, random_state: int = 42):
    """Return train/test MSE across polynomial degrees."""
    X = X_1d.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = []
    for deg in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        pipe = Pipeline([("poly", poly), ("lr", LinearRegression())])
        pipe.fit(X_train, y_train)
        results.append({
            "degree": deg,
            "train_mse": float(mean_squared_error(y_train, pipe.predict(X_train))),
            "test_mse": float(mean_squared_error(y_test, pipe.predict(X_test))),
        })
    return results


def overfit_depth_curve(X, y, task: str = "classification", max_d: int = 20, test_size: float = 0.2, random_state: int = 42):
    """Return train/test score across tree depths."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = []
    for d in range(1, max_d + 1):
        if task == "classification":
            m = DecisionTreeClassifier(max_depth=d, random_state=random_state)
        else:
            m = DecisionTreeRegressor(max_depth=d, random_state=random_state)
        m.fit(X_train, y_train)
        results.append({
            "depth": d,
            "train_score": float(m.score(X_train, y_train)),
            "test_score": float(m.score(X_test, y_test)),
        })
    return results


# ── Classification ────────────────────────────────────────────────────────────

def train_classification(X, y, model_type: str, **kwargs):
    """Train and evaluate any classification model."""
    test_size = kwargs.get("test_size", 0.2)
    random_state = kwargs.get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=kwargs.get("n_neighbors", 5))
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(max_depth=kwargs.get("max_depth", None), random_state=random_state)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            random_state=random_state
        )
    elif model_type == "logistic":
        model = LogisticRegression(C=kwargs.get("C", 1.0), max_iter=1000, random_state=random_state)
    else:
        model = KNeighborsClassifier(n_neighbors=5)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    n_classes = len(np.unique(y))
    avg = "binary" if n_classes == 2 else "macro"

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "confusion_matrix": cm,
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def get_decision_boundary(model, scaler, X_train, X_test, y_train, y_test, resolution: int = 60):
    """Compute decision boundary grid for 2-feature datasets."""
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_s = scaler.transform(grid)
    zz = model.predict(grid_s).reshape(xx.shape)
    return {
        "xx": xx.tolist(),
        "yy": yy.tolist(),
        "zz": zz.tolist(),
        "x_train": X_train[:, 0].tolist(),
        "y_train": X_train[:, 1].tolist(),
        "labels_train": y_train.tolist(),
        "x_test": X_test[:, 0].tolist(),
        "y_test": X_test[:, 1].tolist(),
        "labels_test": y_test.tolist(),
    }
