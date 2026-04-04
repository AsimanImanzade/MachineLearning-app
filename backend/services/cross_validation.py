"""K-Fold Cross-Validation service."""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def run_kfold_cv(X, y, model_type: str = "random_forest", task: str = "regression",
                 n_splits: int = 5, n_estimators: int = 100, random_state: int = 42):
    """Run K-Fold CV and return per-fold train/val scores."""
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state) \
            if task == "regression" else \
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(random_state=random_state) \
            if task == "classification" else None
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_s)):
        X_train, X_val = X_s[train_idx], X_s[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        results.append({
            "fold": fold_idx + 1,
            "train_score": float(model.score(X_train, y_train)),
            "val_score": float(model.score(X_val, y_val)),
        })

    return results
