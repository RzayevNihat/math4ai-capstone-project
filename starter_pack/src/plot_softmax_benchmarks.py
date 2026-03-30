"""
Math4AI Capstone: Softmax Regression Visualization Suite
National AI Center — AI Academy

This module provides high-quality plotting utilities for Model 1 (Softmax):
    1. Training Benchmarks: Loss and Accuracy curves for the Digits dataset.
    2. Optimizer Study: Comparative analysis of SGD, Momentum, and Adam.
    3. Performance Bars: Final test metrics across different optimizers.

All figures are formatted to meet the project's aesthetic standards (§5.1).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# Global Plotting Configuration
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "legend.frameon": False
})

COLORS = {
    "train": "#2563EB", 
    "val": "#DC2626", 
    "sgd": "#5482F4", 
    "momentum": "#4CB372", 
    "adam": "#E6963E"
}

# Directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Core Visualization Functions
# ============================================================

def plot_training_benchmarks():
    """
    Generate Loss and Accuracy curves from the standard training history.
    Saves to: figures/digits_training_history_softmax.png
    """
    path = os.path.join(RESULTS_DIR, "softmax_training_history.npz")
    if not os.path.exists(path):
        return

    data = np.load(path)
    epochs = np.arange(1, len(data["loss_history"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Cross-Entropy Loss
    axes[0].plot(epochs, data["loss_history"], color=COLORS["train"], label="Training Loss")
    if "val_loss_history" in data:
        axes[0].plot(epochs, data["val_loss_history"], color=COLORS["val"], linestyle="--", label="Validation Loss")
    axes[0].set_title("Softmax: Convergence History (Loss)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Panel B: Classification Accuracy
    axes[1].plot(epochs, data["train_acc_history"], color=COLORS["train"], label="Training Acc")
    if "val_acc_history" in data:
        axes[1].plot(epochs, data["val_acc_history"], color=COLORS["val"], linestyle="--", label="Validation Acc")
    axes[1].set_title("Softmax: Accuracy Evolution")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "digits_training_history_softmax.png"))
    plt.close()


def plot_optimizer_comparison():
    """
    Compare SGD, Momentum, and Adam performance using bar charts and curves.
    Standardizes output with the Neural Network track for consistency.
    """
    path = os.path.join(RESULTS_DIR, "softmax_optimizer_study.npz")
    if not os.path.exists(path):
        return

    data = np.load(path)
    optimizers = ["sgd", "momentum", "adam"]

    # 1. Final Accuracy Comparison (Bar Chart)
    plt.figure(figsize=(8, 5))
    acc_values = [data[f"{opt}_acc"] for opt in optimizers]
    bars = plt.bar(optimizers, acc_values, color=[COLORS[opt] for opt in optimizers], width=0.6)
    
    plt.title("Optimizer Performance: Final Test Accuracy")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1.1)

    # Label values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(FIGURES_DIR, "digits_optimizer_accuracy_bar.png"))
    plt.close()

    # 2. Learning Curves across Optimizers
    for metric in ["train_loss", "val_loss"]:
        plt.figure(figsize=(8, 5))
        for opt in optimizers:
            key = f"{opt}_{metric}"
            if key in data:
                plt.plot(data[key], label=opt.upper(), color=COLORS[opt])
        
        plt.title(f"Optimizer Convergence: {metric.replace('_', ' ').title()}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(FIGURES_DIR, f"digits_optimizer_{metric}.png"))
        plt.close()


def main():
    """
    Execute the visualization suite for Softmax Regression.
    """
    # Generate benchmark plots
    plot_training_benchmarks()
    
    # Generate optimizer study plots
    plot_optimizer_comparison()


if __name__ == "__main__":
    main()