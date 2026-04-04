import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_gen import get_classification_data
from utils.visuals import plot_decision_boundary

st.set_page_config(page_title="K-Nearest Neighbors", page_icon="📍", layout="wide")

st.title("📍 K-Nearest Neighbors (KNN)")

# Theory Section
with st.expander("Theory & Formulas", expanded=False):
    st.markdown("### Minkowski Distance")
    st.latex(r"D(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}")
    st.markdown("When `p=2`, it is the Euclidean distance. When `p=1`, it is the Manhattan distance.")
    st.markdown("### Majority Voting")
    st.latex(r"\hat{y} = \text{mode}(\{y^{(1)}, y^{(2)}, \dots, y^{(K)}\})")

# Sidebar
st.sidebar.header("Data & Model Settings")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Synthetic Classification (2D)", "Breast Cancer (Real)", "Synthetic Classification"])

st.sidebar.markdown("### Hyperparameter Playground")
n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 50, 5, 1, help="Low K = Overfitting, High K = Underfitting")
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
p_param = st.sidebar.selectbox("Distance Metric (p)", [2, 1], format_func=lambda x: "Euclidean (p=2)" if x == 2 else "Manhattan (p=1)")

use_cv = st.sidebar.checkbox("Apply K-Fold Cross Validation", value=False)
cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5) if use_cv else 5

# Load Data
X, y = get_classification_data(dataset_choice)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Instantiation
model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p_param)

if use_cv:
    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
    st.subheader(f"Cross Validation ({cv_folds} Folds)")
    st.write(f"Mean Accuracy: {scores.mean():.4f}  |  Std Dev: {scores.std():.4f}")

# Train
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluation
col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Metrics")
    st.write(f"**Accuracy:** {accuracy_score(y_train, y_pred_train):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_train, y_pred_train):.4f}")
with col2:
    st.subheader("Testing Metrics")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred_test):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred_test):.4f}")

if dataset_choice == "Synthetic Classification (2D)":
    st.subheader(f"Decision Boundary (K={n_neighbors})")
    class ModelWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        def predict(self, X_input):
            return self.model.predict(self.scaler.transform(X_input))
            
    wrapper = ModelWrapper(model, scaler)
    fig = plot_decision_boundary(wrapper, X, y, title=f"KNN Boundary (K={n_neighbors})")
    st.pyplot(fig)
else:
    st.info("KNN does not inherently produce feature importance scores.")
