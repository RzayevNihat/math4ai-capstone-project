"""
Math4AI Capstone: Model Calibration Analysis (Track B)
National AI Center — AI Academy

This script evaluates the calibration of the Softmax Regression model by 
generating a Reliability Diagram. It measures how well the predicted 
confidences (maximum probabilities) align with the actual empirical accuracy.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in the path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.softmax_regression import SoftmaxRegression
from utils.data_utils import load_digits
from utils.metrics import confidence_reliability_table

# ============================================================
# Global Visualization Configuration
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": True
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Analysis & Plotting Logic
# ============================================================

def run_calibration_analysis():
    """
    Trains the Softmax model and generates an optimized Reliability Diagram.
    Fixed overlapping bar issue by adjusting width and alignment.
    """
    data_path = os.path.join(BASE_DIR, "data", "digits_data.npz")
    split_path = os.path.join(BASE_DIR, "data", "digits_split_indices.npz")
    
    if not os.path.exists(data_path):
        return

    # --- 1. Model Preparation ---
    dataset = load_digits(data_path, split_path)
    model = SoftmaxRegression(dataset["n_features"], dataset["n_classes"], lr=0.05, reg=1e-5)
    
    model.train(
        dataset["X_train"], dataset["Y_train"], 
        dataset["X_val"], dataset["Y_val"], 
        epochs=100, verbose=False
    )
    
    # --- 2. Calibration Metrics ---
    confidences = model.predict_confidence(dataset["X_test"])
    is_correct = (model.predict(dataset["X_test"]) == dataset["y_test"])
    
    # Using 10 bins for high-resolution calibration analysis
    rel_table = confidence_reliability_table(confidences, is_correct, n_bins=5)

    # --- 3. Optimized Reliability Diagram ---
    plt.figure(figsize=(7, 6))
    
    # Baseline: Perfect calibration
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated", zorder=1)
    
    # Extract metrics from table
    # Filter out empty bins (where count > 0) to avoid plotting zeros
    valid_bins = rel_table[rel_table[:, 1] > 0]
    bin_centers = valid_bins[:, 2]
    bin_accuracies = valid_bins[:, 3]
    
    # PLOT FIX: Adjusted width and added align='center' to prevent overlapping
    plt.bar(
        bin_centers, 
        bin_accuracies, 
        width=0.07,          # Reduced from 0.08 for better spacing
        alpha=0.7, 
        color="#A5D8FF", 
        edgecolor="#1C7ED6", 
        align='center',      # Ensures bars are centered on the mean confidence
        label="Softmax", 
        zorder=2
    )

    # Labeling
    plt.title("Reliability Diagram - Track B (Softmax Calibration)", pad=15)
    plt.xlabel("Confidence (Predicted Probability)")
    plt.ylabel("Accuracy (Empirical Frequency)")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "track_b_reliability_softmax.png"))
    plt.close()

if __name__ == "__main__":
    run_calibration_analysis()