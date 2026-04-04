"""Linear Regression router: Standard, Lasso, Ridge, ElasticNet."""
from fastapi import APIRouter
from models.schemas import TrainRequest
from services.datasets import make_linear_data, make_polynomial_data, get_housing_dataframe, HOUSING_FEATURES
from services.ml_engine import train_regression, overfit_regression_curve
import numpy as np

router = APIRouter(prefix="/api/linear", tags=["Linear Regression"])


@router.get("/dataset")
def get_dataset():
    """Return housing dataset column names and first 5 rows."""
    df = get_housing_dataframe()
    return {
        "features": HOUSING_FEATURES,
        "target": "MedHouseVal",
        "preview": df.head(5).to_dict(orient="records"),
        "shape": list(df.shape),
    }


@router.post("/train")
def train_linear(req: TrainRequest):
    """Train a linear model variant and return metrics + scatter data."""
    X, y = make_linear_data(n_samples=req.n_samples, noise=req.noise_level, random_state=req.random_state)
    result = train_regression(
        X, y,
        model_type=req.model_type,
        alpha=req.alpha,
        l1_ratio=req.l1_ratio,
        degree=1,
        test_size=req.test_size,
        random_state=req.random_state,
    )
    return result


@router.post("/overfit-sandbox")
def overfit_sandbox(req: TrainRequest):
    """Return train/test MSE curve across polynomial degrees."""
    x, y = make_polynomial_data(
        n_samples=req.n_samples,
        noise=req.noise_level,
        random_state=req.random_state,
    )
    curve = overfit_regression_curve(x, y, max_degree=12)

    # Also return the raw scatter
    return {
        "curve": curve,
        "scatter_x": x.tolist(),
        "scatter_y": y.tolist(),
    }
