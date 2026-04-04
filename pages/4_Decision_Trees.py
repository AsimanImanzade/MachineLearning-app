import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_gen import get_classification_data
from utils.visuals import plot_decision_boundary

st.set_page_config(page_title="Decision Trees", page_icon="🌲", layout="wide")

st.title("🌲 Decision Trees")

# Theory Section
with st.expander("Theory & Formulas", expanded=False):
    st.markdown("### Gini Impurity")
    st.latex(r"G = 1 - \sum_{i=1}^{C} p_i^2")
    st.markdown("### Shannon Entropy")
    st.latex(r"H = -\sum_{i=1}^{C} p_i \log_2(p_i)")
    st.markdown("Information Gain is the reduction in entropy or Gini impurity after a dataset is split on an attribute.")

# Sidebar
st.sidebar.header("Data & Model Settings")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Synthetic Classification (2D)", "Breast Cancer (Real)", "Synthetic Classification"])

st.sidebar.markdown("### Hyperparameter Playground")
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
# None means "unlimited depth" but we will use 1-20
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, help="Low depth = Underfitting, High depth = Overfitting")
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

use_cv = st.sidebar.checkbox("Apply K-Fold Cross Validation", value=False)
cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5) if use_cv else 5

feature_threshold = 0.0
if dataset_choice != "Synthetic Classification (2D)":
    st.sidebar.markdown("### Feature Selection")
    feature_threshold = st.sidebar.slider("Drop columns with importance <", 0.0, 0.5, 0.0, 0.01)

# Load Data
X, y = get_classification_data(dataset_choice)

# Feature Selection logic
if "2D" not in dataset_choice and feature_threshold > 0:
    # Quick init fit to get importances
    temp_model = DecisionTreeClassifier(random_state=42).fit(X, y)
    importances = temp_model.feature_importances_
    cols_to_keep = X.columns[importances >= feature_threshold]
    
    st.write(f"**Feature Selection:** Dropping {len(X.columns) - len(cols_to_keep)} columns. Keeping: {len(cols_to_keep)} columns.")
    if len(cols_to_keep) == 0:
        st.error("All features dropped! Please lower the threshold.")
        st.stop()
    X = X[cols_to_keep]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Instantiation
model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

if use_cv:
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
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
    st.subheader(f"Decision Boundary (Max Depth={max_depth})")
    fig = plot_decision_boundary(model, X, y, title=f"Decision Tree Boundary")
    st.pyplot(fig)
else:
    st.subheader("Feature Importances")
    coef_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    coef_df = coef_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(coef_df.set_index('Feature')['Importance'])
