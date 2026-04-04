import streamlit as st

st.set_page_config(
    page_title="Data Science Interactive Learning",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Interactive Data Science Learning Tool")

st.markdown("""
Welcome to the **Interactive Data Science Learning Tool**! 

This educational web application is designed for data science students to visualize and experiment with different machine learning algorithms.

### 👈 Select a model from the sidebar to get started!

### Features
- **Theory & Formulas:** Learn the math behind the models with LaTeX-rendered formulas.
- **Hyperparameter Playground:** Adjust parameters on the fly and see how they affect the model.
- **Overfitting vs Underfitting:** Demystify model complexity through dynamic visualizations.
- **Feature Selection:** Filter out low-importance columns and observe accuracy impact.
- **Cross-Validation:** Quickly evaluate generalization via K-Fold cross validation.

---
**Core Stack used:** Scikit-learn, Pandas, Matplotlib, Seaborn, and Streamlit.
""")
