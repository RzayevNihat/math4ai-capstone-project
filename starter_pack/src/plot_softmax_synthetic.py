"""
Math4AI Capstone: Synthetic Data Visualization for Softmax Regression
National AI Center — AI Academy

This script evaluates and visualizes Softmax Regression on synthetic 2D datasets:
    1. Linear Gaussian: Checks for optimal linear separation.
    2. Moons: Visualizes the limitations of a linear model on non-linear data.

Outputs include Decision Boundary plots and Training Convergence curves, 
formatted specifically for the project's final report.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure project root is in the path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.softmax_regression import SoftmaxRegression
from utils.data_utils import load_synthetic, standardize

# ============================================================
# Global Visualization Configuration
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2
})

# Custom colormaps for decision regions and classes
BG_CMAP = mcolors.ListedColormap(["#c9d9ea", "#d9aeb6"])
CLASS_COLORS = ["#0b3c8c", "#8b002f"]

# Directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Visualization Logic
# ============================================================

def plot_synthetic_results():
    """
    Executes the training and visualization pipeline for synthetic datasets.
    Generates decision boundaries and loss curves for qualitative analysis.
    """
    for dataset_name in ["linear_gaussian", "moons"]:
        data_path = os.path.join(DATA_DIR, f"{dataset_name}.npz")
        
        if not os.path.exists(data_path):
            continue
        
        # --- 1. Data Preparation ---
        data = load_synthetic(data_path)
        
        # Standardize features using the shared utility function
        X_train, X_val, X_test = standardize(
            data["X_train"], data["X_val"], data["X_test"]
        )
        
        # --- 2. Model Training ---
        model = SoftmaxRegression(data["n_features"], data["n_classes"])
        model.train(
            X_train, data["Y_train"], 
            X_val, data["Y_val"], 
            epochs=200, verbose=False
        )
        
        # Consolidate all samples for plotting context
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])

        # --- 3. Decision Boundary Visualization ---
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a dense meshgrid for smooth contour rendering
        resolution = 400 
        x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
        y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), 
            np.linspace(y_min, y_max, resolution)
        )
        
        # Predict class labels across the grid
        grid_input = np.c_[xx.ravel(), yy.ravel()]
        zz = model.predict(grid_input).reshape(xx.shape)
        
        # Render decision regions and the linear separator
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=BG_CMAP, alpha=0.75)
        ax.contour(xx, yy, zz, levels=[0.5], colors=["#d8d8d8"], linewidths=1.5)
        
        # Overlay scatter points with original class styling
        for class_idx, color in enumerate(CLASS_COLORS):
            mask = (y_all == class_idx)
            ax.scatter(
                X_all[mask, 0], X_all[mask, 1], 
                c=color, edgecolors="black", s=45, 
                label=f"Class {class_idx}", zorder=3
            )
        
        title_clean = dataset_name.replace('_', ' ').capitalize()
        ax.set_title(f"Softmax Decision Boundary: {title_clean}", pad=15)
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{dataset_name}_boundary_softmax.png"))
        plt.close()

        # --- 4. Training Convergence Plot ---
        plt.figure(figsize=(6, 4))
        plt.plot(model.train_loss_hist, color="#2563EB", label="Training Loss")
        plt.plot(model.val_loss_hist, color="#DC2626", linestyle="--", label="Validation Loss")
        
        plt.title(f"Convergence Curves: {title_clean}")
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{dataset_name}_curves_softmax.png"))
        plt.close()

if __name__ == "__main__":
    plot_synthetic_results()