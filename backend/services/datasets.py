"""Dataset generators and loaders for all ML modules."""
import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.datasets import (
    make_regression, make_classification, make_moons,
    make_circles, make_blobs, fetch_california_housing
)
from sklearn.preprocessing import StandardScaler


# ── Housing Dataset ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_cached_california_housing() -> pd.DataFrame:
    """Cache the fetch operation to save memory."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.columns = [
        "MedIncome", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"
    ]
    return df

def get_housing_dataframe() -> pd.DataFrame:
    """Return California Housing with renamed, interpretable columns."""
    df = _get_cached_california_housing()
    return df.copy()


HOUSING_FEATURES = [
    "MedIncome", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]


# ── Regression data (1-D, polynomially distorted) ────────────────────────────

def make_polynomial_data(
    n_samples: int = 200,
    noise: float = 0.3,
    degree: int = 1,
    random_state: int = 42,
):
    """Generate 1-D data: y = sin(2x) + noise, useful for overfit sandbox."""
    rng = np.random.RandomState(random_state)
    x = rng.uniform(-3, 3, n_samples)
    y = np.sin(2 * x) + rng.normal(0, noise, n_samples)
    return x, y


def make_linear_data(
    n_samples: int = 200,
    noise: float = 0.3,
    n_features: int = 1,
    random_state: int = 42,
):
    """Simple linear regression data."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise * 50,
        random_state=random_state,
    )
    return X, y


# ── Classification datasets ───────────────────────────────────────────────────

def get_classification_dataset(name: str, noise: float = 0.2, n_samples: int = 300, random_state: int = 42):
    """Return X, y for named classification dataset."""
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise * 5, random_state=random_state)
    else:  # default: iris-like 2-feature
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, random_state=random_state, n_clusters_per_class=1
        )
    return X.astype(float), y.astype(int)
