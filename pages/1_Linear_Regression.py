import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_gen import get_regression_data
from utils.visuals import plot_regression_curve

st.set_page_config(page_title="Linear Regression", page_icon="📈", layout="wide")

st.title("📈 Linear Regression Models")

# Theory Section
with st.expander("Theory & Formulas", expanded=False):
    st.markdown("### Ordinary Least Squares (OLS)")
    st.latex(r"J(\theta) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2")
    st.markdown("### Ridge Regression (L2 Penalty)")
    st.latex(r"J(\theta) = \text{MSE} + \alpha \sum_{j=1}^{p} \theta_j^2")
    st.markdown("### Lasso Regression (L1 Penalty)")
    st.latex(r"J(\theta) = \text{MSE} + \alpha \sum_{j=1}^{p} |\theta_j|")
    st.markdown("### ElasticNet")
    st.latex(r"J(\theta) = \text{MSE} + r \alpha \sum_{j=1}^{p} |\theta_j| + \frac{1-r}{2} \alpha \sum_{j=1}^{p} \theta_j^2")

# Sidebar - Dataset Selection & Hyperparameters
st.sidebar.header("Data & Model Settings")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Synthetic Regression (1D)", "Diabetes (Real)", "Synthetic Regression"])

model_type = st.sidebar.selectbox("Model Type", ["Standard Linear", "Ridge", "Lasso", "ElasticNet"])

# Hyperparameters
alpha = 1.0
l1_ratio = 0.5
poly_degree = 1
use_cv = st.sidebar.checkbox("Apply K-Fold Cross Validation", value=False)
cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5) if use_cv else 5

if model_type in ["Ridge", "Lasso", "ElasticNet"]:
    alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0, 0.01)
if model_type == "ElasticNet":
    l1_ratio = st.sidebar.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.1)

if dataset_choice == "Synthetic Regression (1D)":
    poly_degree = st.sidebar.slider("Polynomial Degree (To show under/overfitting)", 1, 15, 1)

feature_threshold = 0.0
if dataset_choice != "Synthetic Regression (1D)":
    st.sidebar.markdown("### Feature Selection")
    feature_threshold = st.sidebar.slider("Drop columns with abs(coeff) <", 0.0, 50.0, 0.0, 0.5)

# Load Data
X, y = get_regression_data(dataset_choice)

# Feature Selection logic
if "1D" not in dataset_choice and feature_threshold > 0:
    # Quick init fit to get coeffs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    temp_model = LinearRegression().fit(X_scaled, y)
    coeffs = np.abs(temp_model.coef_)
    cols_to_keep = X.columns[coeffs >= feature_threshold]
    
    st.write(f"**Feature Selection:** Dropping {len(X.columns) - len(cols_to_keep)} columns. Keeping: {len(cols_to_keep)} columns.")
    if len(cols_to_keep) == 0:
        st.error("All features dropped! Please lower the threshold.")
        st.stop()
    X = X[cols_to_keep]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for 1D Polynomials
if dataset_choice == "Synthetic Regression (1D)" and poly_degree > 1:
    poly = PolynomialFeatures(degree=poly_degree)
    X_train_processed = poly.fit_transform(X_train)
    X_test_processed = poly.transform(X_test)
    X_full_processed = poly.transform(X)
else:
    # Scale real datasets
    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train)
    X_test_processed = scaler.transform(X_test)
    X_full_processed = scaler.transform(X)

# Model Instantiation
if model_type == "Standard Linear":
    model = LinearRegression()
elif model_type == "Ridge":
    model = Ridge(alpha=alpha)
elif model_type == "Lasso":
    model = Lasso(alpha=alpha)
elif model_type == "ElasticNet":
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

# CV
if use_cv:
    scores = cross_val_score(model, X_full_processed, y, cv=cv_folds, scoring='neg_mean_squared_error')
    st.subheader(f"Cross Validation ({cv_folds} Folds)")
    st.write(f"Mean MSE: {-scores.mean():.4f}  |  Std Dev: {scores.std():.4f}")

# Train
model.fit(X_train_processed, y_train)

# Evaluation
y_pred_train = model.predict(X_train_processed)
y_pred_test = model.predict(X_test_processed)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Metrics")
    st.write(f"**MSE:** {mean_squared_error(y_train, y_pred_train):.4f}")
    st.write(f"**R²:** {r2_score(y_train, y_pred_train):.4f}")
with col2:
    st.subheader("Testing Metrics")
    st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_test):.4f}")
    st.write(f"**R²:** {r2_score(y_test, y_pred_test):.4f}")

# Visuals
if dataset_choice == "Synthetic Regression (1D)":
    st.subheader(f"Overfitting vs Underfitting (Degree: {poly_degree})")
    
    # We need a wrapper to predict directly from 1D X since plot func assumes raw X
    class ModelWrapper:
        def __init__(self, model, poly, degree):
            self.model = model
            self.poly = poly
            self.degree = degree
        def predict(self, X_input):
            if self.degree > 1:
                return self.model.predict(self.poly.transform(X_input))
            return self.model.predict(X_input)
            
    wrapper = ModelWrapper(model, poly if poly_degree > 1 else None, poly_degree)
    fig = plot_regression_curve(wrapper, X.values, y.values, title=f"{model_type} Fit")
    st.pyplot(fig)
else:
    # Feature Importance for real models
    st.subheader("Feature Coefficients (Importance)")
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False)
    st.bar_chart(coef_df.set_index('Feature')['Coefficient'])
