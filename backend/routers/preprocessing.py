"""Preprocessing pipeline router: Outlier handling, Feature Scaling, Categorical encoding."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import os
from functools import lru_cache
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

router = APIRouter(prefix="/api/linear", tags=["Preprocessing"])

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "housing_prices.csv")


@lru_cache(maxsize=1)
def __cached_load_housing():
    return pd.read_csv(DATA_PATH)

def _load_housing():
    """Load the housing prices dataset."""
    df = __cached_load_housing()
    return df.copy()


class PreprocessRequest(BaseModel):
    scaler_type: str = "none"  # none | minmax | maxabs | standard | robust | log
    outlier_features: List[str] = []  # which numeric features to apply IQR outlier removal
    test_size: float = 0.2
    random_state: int = 42
    selected_features: Optional[List[str]] = None


@router.get("/preprocessing/dataset-info")
def dataset_info():
    """Return dataset overview."""
    df = _load_housing()
    
    # Generate correlation matrix of encoded features for frontend
    corr_df = df.copy()
    ordinal_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    corr_df["furnishingstatus"] = corr_df["furnishingstatus"].replace(ordinal_map)
    binary_nominal = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    corr_df = pd.get_dummies(corr_df, columns=binary_nominal, drop_first=True, dtype=int)
    
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    corr_matrix_raw = corr_df.corr(numeric_only=True).round(2)
    corr_matrix = []
    for col in corr_matrix_raw.columns:
        row = {"feature": col}
        for inner_col in corr_matrix_raw.columns:
            row[inner_col] = float(corr_matrix_raw.loc[col, inner_col])
        corr_matrix.append(row)
        
    encoded_features = [c for c in corr_df.columns if c != "price"]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    unique_values = {}
    for col in df.columns:
        uvals = df[col].unique().tolist()
        if len(uvals) > 20:
            unique_values[col] = {"count": len(uvals), "sample": uvals[:10], "truncated": True}
        else:
            unique_values[col] = {"count": len(uvals), "sample": uvals, "truncated": False}

    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "preview": df.head(10).to_dict(orient="records"),
        "describe": df.describe().round(2).to_dict(),
        "unique_values": unique_values,
        "correlation_matrix": corr_matrix,
        "encoded_features": encoded_features,
    }


@router.post("/preprocessing/train")
def preprocessing_train(req: PreprocessRequest):
    """Full preprocessing pipeline: outlier removal → categorical encoding → scaling → train."""
    df = _load_housing()

    target = "price"
    numeric_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
    binary_nominal = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    ordinal_col = "furnishingstatus"

    # ── Step 1: Outlier detection ─────────────────────────────────
    outlier_info = {}
    rows_before = len(df)
    for feat in req.outlier_features:
        if feat in df.columns and (feat in numeric_cols or feat == target):
            Q1 = df[feat].quantile(0.25)
            Q3 = df[feat].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (df[feat] < lower) | (df[feat] > upper)
            n_outliers = int(mask.sum())
            outlier_info[feat] = {
                "Q1": round(float(Q1), 2),
                "Q3": round(float(Q3), 2),
                "IQR": round(float(IQR), 2),
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
                "n_outliers": n_outliers,
            }
            df = df[~mask]
    rows_after = len(df)

    # ── Step 2: Train/Test Split ──────────────────────────────────
    X = df.drop(columns=[target])
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=req.random_state
    )

    # ── Step 3: Categorical Encoding ──────────────────────────────
    # Ordinal: furnishingstatus
    ordinal_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    X_train[ordinal_col] = X_train[ordinal_col].replace(ordinal_map)
    X_test[ordinal_col] = X_test[ordinal_col].replace(ordinal_map)

    # Nominal: binary yes/no columns → dummy (drop_first=True)
    X_train_dummies = pd.get_dummies(X_train[binary_nominal], dtype=int, drop_first=True)
    X_test_dummies = pd.get_dummies(X_test[binary_nominal], dtype=int, drop_first=True)

    # Ensure same columns
    for col in X_train_dummies.columns:
        if col not in X_test_dummies.columns:
            X_test_dummies[col] = 0
    X_test_dummies = X_test_dummies[X_train_dummies.columns]

    X_train = X_train.drop(columns=binary_nominal)
    X_test = X_test.drop(columns=binary_nominal)

    X_train = pd.concat([X_train.reset_index(drop=True), X_train_dummies.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), X_test_dummies.reset_index(drop=True)], axis=1)

    # Convert all to numeric
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_names_after_encoding = list(X_train.columns)
    preview_after_encoding_train = X_train.head(5).round(4).to_dict(orient="records")

    all_available_features = list(X_train.columns)

    if req.selected_features is not None:
        valid_features = [f for f in req.selected_features if f in X_train.columns]
        if valid_features:
            X_train = X_train[valid_features]
            X_test = X_test[valid_features]

    # ── Step 4: Feature Scaling ───────────────────────────────────
    scaler_name = req.scaler_type
    scaler_used = "None"
    numeric_to_scale = [c for c in numeric_cols if c in X_train.columns]

    if scaler_name == "log":
        for c in numeric_to_scale:
            X_train[c] = np.log1p(X_train[c].clip(lower=0))
            X_test[c] = np.log1p(X_test[c].clip(lower=0))
        scaler_used = "Log Transform (np.log1p)"
    elif scaler_name != "none":
        scalers = {
            "minmax": MinMaxScaler(),
            "maxabs": MaxAbsScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }
        scaler = scalers.get(scaler_name)
        if scaler:
            X_train[numeric_to_scale] = scaler.fit_transform(X_train[numeric_to_scale])
            X_test[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])
            scaler_used = scaler.__class__.__name__

    preview_after_scaling_train = X_train.head(5).round(4).to_dict(orient="records")

    # ── Step 5: Train Model ───────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train.values, y_train)
    y_pred_train = model.predict(X_train.values)
    y_pred_test = model.predict(X_test.values)

    def calc_metrics(y_true, y_pred):
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse": round(mse, 2),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "rmse": round(float(np.sqrt(mse)), 2),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }

    train_metrics = calc_metrics(y_train, y_pred_train)
    test_metrics = calc_metrics(y_test, y_pred_test)

    # Coefficients
    coefs = []
    for i, f in enumerate(feature_names_after_encoding):
        coefs.append({"feature": str(f), "value": round(float(model.coef_[i]), 2)})

    # Actual vs Predicted scatter (test, limit 200)
    n_show = min(200, len(y_test))
    scatter_actual = y_test[:n_show].tolist()
    scatter_predicted = y_pred_test[:n_show].tolist()

    # ── Code Steps ────────────────────────────────────────────────
    code_steps = [
        {
            "title": "1. Import Libraries",
            "code": "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler"
        },
        {
            "title": "2. Load Dataset",
            "code": "df = pd.read_csv('housing.csv')\nprint(df.shape)\nprint(df.head(10))\nprint(df.dtypes)"
        },
        {
            "title": "3. Explore Unique Values",
            "code": "# Check unique values for each column\nfor col in df.columns:\n    print(f'{col}: {df[col].nunique()} unique values')\n    print(f'  Values: {df[col].unique()[:10]}\\n')"
        },
        {
            "title": "4. Outlier Detection (IQR)",
            "code": "def outlier_detect(x):\n    Q1 = x.quantile(0.25)\n    Q3 = x.quantile(0.75)\n    IQR = Q3 - Q1\n    return ((x < Q1 - 1.5 * IQR) | (x > Q3 + 1.5 * IQR))\n\n# Apply to numeric columns\nfor col in ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']:\n    mask = outlier_detect(df[col])\n    print(f'{col}: {mask.sum()} outliers found')\n    df = df[~mask]\nprint(f'Shape after outlier removal: {df.shape}')"
        },
        {
            "title": "5. Train-Test Split",
            "code": f"X = df.drop(columns=['price'])\ny = df['price']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size={req.test_size}, random_state=42\n)\nprint(f'Train: {{X_train.shape}}, Test: {{X_test.shape}}')"
        },
        {
            "title": "6. Ordinal Encoding",
            "code": "# furnishingstatus is ordinal: unfurnished < semi-furnished < furnished\nX_train['furnishingstatus'] = X_train['furnishingstatus'].replace(\n    {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}\n)\nX_test['furnishingstatus'] = X_test['furnishingstatus'].replace(\n    {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}\n)\nprint(X_train['furnishingstatus'].value_counts())"
        },
        {
            "title": "7. Nominal Encoding (Dummy / OneHotEncoding)",
            "code": "# Binary nominal columns: yes/no → 1/0 (drop_first=True)\nncf = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',\n       'airconditioning', 'prefarea']\n\nX_train = X_train.merge(\n    pd.get_dummies(X_train[ncf], dtype=int, drop_first=True),\n    left_index=True, right_index=True\n)\nX_train = X_train.drop(columns=ncf)\n\nX_test = X_test.merge(\n    pd.get_dummies(X_test[ncf], dtype=int, drop_first=True),\n    left_index=True, right_index=True\n)\nX_test = X_test.drop(columns=ncf)\nprint(X_train.head())"
        },
        {
            "title": "8. Feature Scaling",
            "code": f"# Apply {scaler_used}\nnumeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']\n\n" + (
                "# Log Transform\nfor col in numeric_cols:\n    X_train[col] = np.log1p(X_train[col])\n    X_test[col] = np.log1p(X_test[col])" if scaler_name == "log" else
                f"scaler = {scaler_used}()\nX_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])\nX_test[numeric_cols] = scaler.transform(X_test[numeric_cols])" if scaler_name != "none" else
                "# No scaling applied"
            ) + "\nprint(X_train.head())"
        },
        {
            "title": "9. Train & Evaluate",
            "code": f"model = LinearRegression()\nmodel.fit(X_train, y_train)\n\ny_pred_train = model.predict(X_train)\ny_pred_test = model.predict(X_test)\n\nprint(f'Train R²:  {{r2_score(y_train, y_pred_train):.4f}}')\nprint(f'Test  R²:  {{r2_score(y_test, y_pred_test):.4f}}')\nprint(f'Train RMSE: {{np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}}')\nprint(f'Test  RMSE: {{np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}}')"
        },
    ]

    return {
        "rows_before_outlier": rows_before,
        "rows_after_outlier": rows_after,
        "outlier_info": outlier_info,
        "scaler_used": scaler_used,
        "feature_names_after_encoding": feature_names_after_encoding,
        "preview_after_encoding": preview_after_encoding_train,
        "preview_after_scaling": preview_after_scaling_train,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "coefficients": coefs,
        "intercept": round(float(model.intercept_), 2),
        "scatter_actual": scatter_actual,
        "scatter_predicted": scatter_predicted,
        "code_steps": code_steps,
        "all_available_features": all_available_features,
    }
