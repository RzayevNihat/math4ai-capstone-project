"""
Math4AI Capstone: Track B - Reliability Diagram for One-Hidden-Layer Neural Network
National AI Center — AI Academy

This script trains the final NN (hidden=32, Adam) and generates the Reliability Diagram
for Track B, exactly matching the style of the Softmax version.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_nn import train_nn
from nn_model import OneHiddenLayerNN
from utils.data_utils import load_digits
from utils.metrics import confidence_reliability_table

# ============================================================
# Aesthetic Configuration (same as other plots)
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
# Main Function
# ============================================================
def run_track_b_nn():
    print("=== Running Track B for Neural Network ===")
    
    # 1. Load data (same fixed splits as all other scripts)
    data_path = os.path.join(BASE_DIR, "data", "digits_data.npz")
    split_path = os.path.join(BASE_DIR, "data", "digits_split_indices.npz")
    dataset = load_digits(data_path, split_path)

    # 2. Train final NN (exact same config as your repeated seeds)
    model, _, best_epoch = train_nn(
        X_train=dataset["X_train"],
        y_train=dataset["y_train"],
        X_val=dataset["X_val"],
        y_val=dataset["y_val"],
        input_dim=dataset["n_features"],
        hidden_dim=32,
        output_dim=dataset["n_classes"],
        optimizer="adam",
        lr=0.001,
        reg_lambda=1e-4,
        batch_size=64,
        epochs=200,
        seed=42,
        checkpoint_policy="best_val",
        verbose=False,
    )

    print(f"NN trained (best epoch: {best_epoch})")

    # 3. Get predictions on test set
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    probs = model.forward(X_test)                    # shape (N, 10)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)              # max probability = confidence
    is_correct = (preds == y_test).astype(float)

    # 4. Compute reliability table (same function used for softmax)
    rel_table = confidence_reliability_table(confidences, is_correct, n_bins=5)

    # 5. Plot Reliability Diagram
    plt.figure(figsize=(7, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated", zorder=1)
    
    # Filter empty bins
    valid_bins = rel_table[rel_table[:, 1] > 0]
    bin_centers = valid_bins[:, 2]
    bin_accuracies = valid_bins[:, 3]

    # Bar plot for NN
    plt.bar(
        bin_centers,
        bin_accuracies,
        width=0.07,
        alpha=0.7,
        color="#FF7F0E",          # orange color (matches NN theme)
        edgecolor="#D62728",
        align='center',
        label="One-Hidden-Layer NN",
        zorder=2
    )

    plt.title("Reliability Diagram - Track B (One-Hidden-Layer NN)", pad=15)
    plt.xlabel("Confidence (Predicted Probability)")
    plt.ylabel("Accuracy (Empirical Frequency)")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper left")
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "track_b_reliability_nn.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Track B plot for Neural Network saved:\n   {save_path}")
    print("   (You can now add this figure to your LaTeX report)")

if __name__ == "__main__":
    run_track_b_nn()