import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_gen import get_classification_data
from utils.visuals import plot_decision_boundary

st.set_page_config(page_title="Logistic Regression", page_icon="🔗", layout="wide")

st.title("🔗 Logistic Regression")

# Theory Section
with st.expander("Theory & Formulas", expanded=False):
    st.markdown("### Sigmoid Function")
    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
    st.markdown("### Log Loss (Cross-Entropy)")
    st.latex(r"J(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")
    st.markdown("### Regularization (L2 Penalty)")
    st.latex(r"J(\theta) = \text{LogLoss} + \frac{1}{C} \sum_{j=1}^{p} \theta_j^2")

# Sidebar
st.sidebar.header("Data & Model Settings")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Synthetic Classification (2D)", "Breast Cancer (Real)", "Synthetic Classification"])

penalty = st.sidebar.selectbox("Penalty", ["l2", "l1"])
C_param = st.sidebar.slider("C (Inverse Regularization Strength)", 0.01, 10.0, 1.0, 0.01)

use_cv = st.sidebar.checkbox("Apply K-Fold Cross Validation", value=False)
cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5) if use_cv else 5

# Load Data
X, y = get_classification_data(dataset_choice)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Instantiation
solver = 'saga' if penalty == 'l1' else 'lbfgs'
model = LogisticRegression(penalty=penalty, C=C_param, solver=solver, max_iter=2000)

if use_cv:
    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
    st.subheader(f"Cross Validation ({cv_folds} Folds)")
    st.write(f"Mean Accuracy: {scores.mean():.4f}  |  Std Dev: {scores.std():.4f}")

# Train
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_prod_train = model.predict_proba(X_train)
y_pred_test = model.predict(X_test)
y_pred_prod_test = model.predict_proba(X_test)

# Evaluation
col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Metrics")
    st.write(f"**Accuracy:** {accuracy_score(y_train, y_pred_train):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_train, y_pred_train):.4f}")
    st.write(f"**Log Loss:** {log_loss(y_train, y_pred_prod_train):.4f}")
with col2:
    st.subheader("Testing Metrics")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred_test):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred_test):.4f}")
    st.write(f"**Log Loss:** {log_loss(y_test, y_pred_prod_test):.4f}")

if dataset_choice == "Synthetic Classification (2D)":
    st.subheader("Decision Boundary")
    class ModelWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        def predict(self, X_input):
            return self.model.predict(self.scaler.transform(X_input))
            
    wrapper = ModelWrapper(model, scaler)
    fig = plot_decision_boundary(wrapper, X, y, title=f"Logistic Regression Boundary (C={C_param})")
    st.pyplot(fig)
else:
    st.subheader("Feature Coefficients (Importance)")
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False)
    st.bar_chart(coef_df.set_index('Feature')['Coefficient'])
