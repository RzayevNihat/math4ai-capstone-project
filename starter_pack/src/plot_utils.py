import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_training_curves(results, save_dir="starter_pack/figures"):
    """
    results: list of dicts loaded from digits_optimizer_study.json
    """
    ensure_dir(save_dir)

    # -------- Validation Loss --------
    plt.figure(figsize=(8, 5))
    for r in results:
        epochs = np.arange(1, len(r["history"]["val_loss"]) + 1)
        plt.plot(epochs, r["history"]["val_loss"], label=r["optimizer"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross-Entropy Loss")
    plt.title("Digits Optimizer Study - Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "digits_optimizer_val_loss.png", dpi=200)
    plt.close()

    # -------- Validation Accuracy --------
    plt.figure(figsize=(8, 5))
    for r in results:
        epochs = np.arange(1, len(r["history"]["val_acc"]) + 1)
        plt.plot(epochs, r["history"]["val_acc"], label=r["optimizer"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Digits Optimizer Study - Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "digits_optimizer_val_acc.png", dpi=200)
    plt.close()

    # -------- Train Loss --------
    plt.figure(figsize=(8, 5))
    for r in results:
        epochs = np.arange(1, len(r["history"]["train_loss"]) + 1)
        plt.plot(epochs, r["history"]["train_loss"], label=r["optimizer"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Cross-Entropy Loss")
    plt.title("Digits Optimizer Study - Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "digits_optimizer_train_loss.png", dpi=200)
    plt.close()

    # -------- Train Accuracy --------
    plt.figure(figsize=(8, 5))
    for r in results:
        epochs = np.arange(1, len(r["history"]["train_acc"]) + 1)
        plt.plot(epochs, r["history"]["train_acc"], label=r["optimizer"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Digits Optimizer Study - Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "digits_optimizer_train_acc.png", dpi=200)
    plt.close()


def plot_optimizer_summary(results, save_dir="starter_pack/figures"):
    ensure_dir(save_dir)

    optimizers = [r["optimizer"] for r in results]
    test_acc = [r["test_accuracy"] for r in results]
    test_loss = [r["test_loss"] for r in results]
    val_acc = [r["val_accuracy"] for r in results]
    val_loss = [r["val_loss"] for r in results]

    x = np.arange(len(optimizers))

    # -------- Accuracy bar plot --------
    plt.figure(figsize=(7, 5))
    plt.bar(x, val_acc, width=0.4, label="Validation Accuracy")
    plt.bar(x + 0.4, test_acc, width=0.4, label="Test Accuracy")
    plt.xticks(x + 0.2, optimizers)
    plt.ylabel("Accuracy")
    plt.title("Optimizer Comparison - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "digits_optimizer_accuracy_bar.png", dpi=200)
    plt.close()

    # -------- Loss bar plot --------
    plt.figure(figsize=(7, 5))
    plt.bar(x, val_loss, width=0.4, label="Validation Loss")
    plt.bar(x + 0.4, test_loss, width=0.4, label="Test Loss")
    plt.xticks(x + 0.2, optimizers)
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Optimizer Comparison - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "digits_optimizer_loss_bar.png", dpi=200)
    plt.close()