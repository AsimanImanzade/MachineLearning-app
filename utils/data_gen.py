import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, make_regression, make_classification

def get_regression_data(dataset_name="Diabetes (Real)"):
    if dataset_name == "Diabetes (Real)":
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        return X, y
    elif dataset_name == "Synthetic Regression":
        X, y = make_regression(n_samples=300, n_features=10, noise=20.0, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        return X, y
    elif dataset_name == "Synthetic Regression (1D)":
        # Useful for easily visualizing under/overfitting in 2D plots
        X, y = make_regression(n_samples=200, n_features=1, noise=15.0, random_state=42)
        # add some non-linearity for polynomial regression examples
        y = y + X[:, 0]**3 * 20
        X = pd.DataFrame(X, columns=["feature_0"])
        y = pd.Series(y, name='target')
        return X, y
    return None, None

def get_classification_data(dataset_name="Breast Cancer (Real)"):
    if dataset_name == "Breast Cancer (Real)":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        return X, y
    elif dataset_name == "Synthetic Classification":
        X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        return X, y
    elif dataset_name == "Synthetic Classification (2D)":
        # Useful for decision boundaries
        X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, 
                                   n_clusters_per_class=1, flip_y=0.1, class_sep=1.0, random_state=42)
        feature_names = [f"feature_{i}" for i in range(2)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        return X, y
    return None, None
