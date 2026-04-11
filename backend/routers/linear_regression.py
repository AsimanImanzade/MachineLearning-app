"""Linear Regression router: Standard, Lasso, Ridge, ElasticNet."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from models.schemas import TrainRequest
from services.datasets import make_linear_data, make_polynomial_data, get_housing_dataframe, HOUSING_FEATURES
from services.ml_engine import train_regression, overfit_regression_curve
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

router = APIRouter(prefix="/api/linear", tags=["Linear Regression"])


class RealDatasetRequest(BaseModel):
    features: Optional[List[str]] = None
    test_size: float = 0.2


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


@router.post("/real-dataset")
def real_dataset(req: RealDatasetRequest):
    """Train on real California Housing dataset and return comprehensive results."""
    df = get_housing_dataframe()
    target = "MedHouseVal"

    features = req.features if req.features else HOUSING_FEATURES

    X = df[features].values
    y = df[target].values

    # Correlation matrix
    corr_df = df[features + [target]].corr()
    corr_matrix = {
        "columns": list(corr_df.columns),
        "values": corr_df.values.tolist(),
    }

    # DataFrame preview (first 10 rows)
    preview = df[features + [target]].head(10).to_dict(orient="records")
    describe = df[features + [target]].describe().to_dict()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def calc_metrics(y_true, y_pred):
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse": round(mse, 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }

    train_metrics = calc_metrics(y_train, y_pred_train)
    test_metrics = calc_metrics(y_test, y_pred_test)

    # Coefficients
    coefs = []
    for i, f in enumerate(features):
        coefs.append({"feature": f, "value": round(float(model.coef_[i]), 4)})

    # Scatter: actual vs predicted for test set (limit to 200 for bandwidth)
    n_show = min(200, len(y_test))
    scatter_actual = y_test[:n_show].tolist()
    scatter_predicted = y_pred_test[:n_show].tolist()

    # Best feature scatter (MedIncome is typically the strongest predictor)
    best_feature_idx = 0  # MedIncome
    if "MedIncome" in features:
        best_feature_idx = features.index("MedIncome")

    best_feature_x_train = X_train[:, best_feature_idx].tolist()
    best_feature_x_test = X_test[:n_show, best_feature_idx].tolist()

    # Simple regression on best feature for the line
    from sklearn.linear_model import LinearRegression as LR
    simple_model = LR()
    simple_model.fit(X_train[:, best_feature_idx:best_feature_idx+1], y_train)
    x_line = np.linspace(
        X_train[:, best_feature_idx].min(),
        X_train[:, best_feature_idx].max(),
        50
    )
    y_line = simple_model.predict(x_line.reshape(-1, 1))

    # Python code for students
    code_steps = [
        {
            "title": "1. Import Libraries",
            "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.datasets import fetch_california_housing"
        },
        {
            "title": "2. Load Dataset",
            "code": "housing = fetch_california_housing(as_frame=True)\ndf = housing.frame\nprint(df.head())\nprint(f'Shape: {df.shape}')"
        },
        {
            "title": "3. Explore Data",
            "code": "# Check for missing values\nprint(df.isnull().sum())\n\n# Summary statistics\nprint(df.describe())\n\n# Correlation matrix\ncorrelation = df.corr()\nprint(correlation['MedHouseVal'].sort_values(ascending=False))"
        },
        {
            "title": "4. Prepare Features & Target",
            "code": f"features = {features}\nX = df[features]\ny = df['MedHouseVal']"
        },
        {
            "title": "5. Train-Test Split",
            "code": f"X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size={req.test_size}, random_state=42\n)\nprint(f'Train size: {{X_train.shape[0]}}')\nprint(f'Test size:  {{X_test.shape[0]}}')"
        },
        {
            "title": "6. Train the Model",
            "code": "model = LinearRegression()\nmodel.fit(X_train, y_train)\n\nprint('Coefficients:')\nfor feat, coef in zip(features, model.coef_):\n    print(f'  {feat}: {coef:.4f}')\nprint(f'Intercept: {model.intercept_:.4f}')"
        },
        {
            "title": "7. Evaluate",
            "code": "y_pred_train = model.predict(X_train)\ny_pred_test = model.predict(X_test)\n\nprint(f'Train R²:  {r2_score(y_train, y_pred_train):.4f}')\nprint(f'Test  R²:  {r2_score(y_test, y_pred_test):.4f}')\nprint(f'Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}')\nprint(f'Test  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}')"
        },
    ]

    return {
        "features": features,
        "target": target,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "preview": preview,
        "describe": describe,
        "correlation": corr_matrix,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "coefficients": coefs,
        "intercept": round(float(model.intercept_), 4),
        "scatter_actual": scatter_actual,
        "scatter_predicted": scatter_predicted,
        "best_feature_name": features[best_feature_idx] if features else "MedIncome",
        "best_feature_x_test": best_feature_x_test,
        "best_feature_line_x": x_line.tolist(),
        "best_feature_line_y": y_line.tolist(),
        "code_steps": code_steps,
    }
