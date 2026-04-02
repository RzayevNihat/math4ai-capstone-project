"""
Math4AI Capstone: Track B — Prediction Confidence and Reliability
National AI Center — Academy

This module performs uncertainty and calibration analysis for Model 1 & 2.
It follows the project requirements (§8.2) for:
    - Confidence: max(p_i) for each sample.
    - Predictive Entropy: H(p) = -sum(p_j * log(p_j)).
    - Calibration: 5-bin reliability diagrams and ECE metrics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Environment & Path Configuration
# ============================================================

_SRC = os.path.dirname(os.path.abspath(__file__))
_SP = os.path.dirname(_SRC)
sys.path.extend([_SRC, os.path.join(_SRC, "models"), os.path.join(_SRC, "utils")])

from data_utils import load_digits
from softmax_regression import SoftmaxRegression
from train_nn import train_nn as train_model_nn

# Directory setup
FIGURES_DIR = os.path.join(_SP, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Aesthetic constants (Matching NN report style)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.25, "legend.frameon": False
})

COLORS = {"sm": "#378ADD", "nn": "#1D9E75", "corr": "#639922", "err": "#E24B4A"}

# ============================================================
# Core Track B Metrics (§8.2 Logic)
# ============================================================

def get_metrics(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence and entropy for a given probability distribution.
    Entropy is clipped to 1e-15 to prevent numerical log(0) errors.
    """
    conf = np.max(probs, axis=1)
    safe_p = np.clip(probs, 1e-15, 1.0)
    ent = -np.sum(safe_p * np.log(safe_p), axis=1)
    return conf, ent


def get_reliability_table(conf: np.ndarray, correct: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Partition predictions into confidence bins to compute calibration.
    Returns: [bin_lo, bin_hi, mean_confidence, empirical_accuracy].
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins-1 else (conf >= lo) & (conf < hi)
        
        if np.any(mask):
            rows.append([lo, hi, conf[mask].mean(), correct[mask].mean()])
        else:
            rows.append([lo, hi, np.nan, np.nan])
    return np.array(rows)

# ============================================================
# Visualization Helpers
# ============================================================

def plot_reliability_diagrams(tbl_sm, tbl_nn, save_path):
    """
    Create side-by-side reliability diagrams for Model 1 and Model 2.
    Includes Expected Calibration Error (ECE) for quantitative comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    
    for ax, tbl, name, color in [(axes[0], tbl_sm, "Softmax Regression", COLORS["sm"]),
                                 (axes[1], tbl_nn, "One-Hidden-Layer NN", COLORS["nn"])]:
        valid = tbl[~np.isnan(tbl[:, 2])]
        ax.bar(valid[:, 2], valid[:, 3], width=0.15, alpha=0.7, color=color, edgecolor="white", label="Empirical Accuracy")
        ax.plot([0, 1], [0, 1], "--", color="#888780", label="Perfect Calibration")
        
        # ECE Calculation: mean absolute difference between confidence and accuracy
        ece = np.nanmean(np.abs(valid[:, 2] - valid[:, 3]))
        ax.text(0.95, 0.05, f"ECE: {ece:.4f}", transform=ax.transAxes, ha="right", fontweight="bold")
        
        ax.set_title(name)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1.05)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("Empirical Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_uncertainty_dist(val_sm, val_nn, corr_sm, corr_nn, save_path, mode="Confidence"):
    """
    Visualize density of metrics for correct vs incorrect predictions.
    Helps identify if the model is 'overconfident' in its errors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    max_x = np.log(10) if mode == "Entropy" else 1.0
    bins = np.linspace(0, max_x, 20)

    for ax, val, corr, name in [(axes[0], val_sm, corr_sm, "Softmax Regression"),
                                (axes[1], val_nn, corr_nn, "One-Hidden-Layer NN")]:
        ax.hist(val[corr], bins=bins, alpha=0.5, color=COLORS["corr"], label="Correct", density=True)
        ax.hist(val[~corr], bins=bins, alpha=0.5, color=COLORS["err"], label="Incorrect", density=True)
        
        ax.axvline(val[corr].mean(), color=COLORS["corr"], linestyle="--", label=f"Mean Correct: {val[corr].mean():.2f}")
        ax.axvline(val[~corr].mean(), color=COLORS["err"], linestyle="--", label=f"Mean Incorrect: {val[~corr].mean():.2f}")
        
        ax.set_title(name)
        ax.set_xlabel(f"Predictive {mode}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================================================
# Main Execution Pipeline
# ============================================================

def main():
    # 1. Load benchmark dataset
    ds = load_digits(os.path.join(_SP, "data", "digits_data.npz"), 
                     os.path.join(_SP, "data", "digits_split_indices.npz"))
    
    # 2. Model 1 (Softmax) Training & Best Weight Restoration
    sm = SoftmaxRegression(ds["n_features"], ds["n_classes"], lr=0.05, reg=1e-4)
    sm.train(ds["X_train"], ds["Y_train"], ds["X_val"], ds["Y_val"], epochs=200, verbose=False)
    sm.load_best_weights()

    # 3. Model 2 (NN) Training
    nn, _, _ = train_model_nn(ds["X_train"], ds["y_train"], ds["X_val"], ds["y_val"],
                              input_dim=ds["n_features"], hidden_dim=32, 
                              output_dim=ds["n_classes"], epochs=200, verbose=False)

    # 4. Probabilistic Analysis
    p_sm = sm.forward_pass(ds["X_test"])
    p_nn = nn.predict_proba(ds["X_test"])

    conf_sm, ent_sm = get_metrics(p_sm)
    conf_nn, ent_nn = get_metrics(p_nn)

    # Boolean mask for correctness
    corr_sm = (np.argmax(p_sm, axis=1) == ds["y_test"])
    corr_nn = (np.argmax(p_nn, axis=1) == ds["y_test"])

    # 5. Generate Figures
    plot_reliability_diagrams(get_reliability_table(conf_sm, corr_sm),
                              get_reliability_table(conf_nn, corr_nn),
                              os.path.join(FIGURES_DIR, "track_b_reliability_both.png"))

    plot_uncertainty_dist(conf_sm, conf_nn, corr_sm, corr_nn, 
                          os.path.join(FIGURES_DIR, "track_b_confidence_correctness.png"), "Confidence")
    
    plot_uncertainty_dist(ent_sm, ent_nn, corr_sm, corr_nn, 
                          os.path.join(FIGURES_DIR, "track_b_entropy_correctness.png"), "Entropy")

if __name__ == "__main__":
    main()