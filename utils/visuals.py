import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_regression_curve(model, X, y, title="Regression Fit"):
    """
    Plots the regression fit for a 1D dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    
    # Sort X for a smooth line
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    
    if hasattr(model, 'predict'):
        y_plot = model.predict(X_plot)
        ax.plot(X_plot, y_plot, color='red', linewidth=2, label='Model Fit')
    
    ax.set_title(title)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Target")
    ax.legend()
    return fig

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plots the decision boundary for a 2D dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a mesh grid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # Predict over the grid
    if hasattr(model, 'predict'):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot filled contour
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # Plot real points
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    return fig
