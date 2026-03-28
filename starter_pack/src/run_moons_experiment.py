import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_nn import train_nn


def load_moons():
    data = np.load("starter_pack/data/moons.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_train_val_test_split(X, y, train_frac=0.6, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def evaluate(model, X, y):
    probs = model.forward(X)
    loss = model.compute_loss(y)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "preds": preds,
        "probs": probs,
    }


def plot_decision_boundary(model, X, y, title, save_path, h=0.01):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid)
    preds = np.argmax(probs, axis=1)
    zz = preds.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, alpha=0.35)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=25)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_curves(history, title, save_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_moons()
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.hstack([y_train, y_val, y_test])
    figures_dir = Path("starter_pack/figures")
    results_dir = Path("starter_pack/results")
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    hidden_widths = [2, 8, 32]
    summary = []

    for hidden_dim in hidden_widths:
        print("=" * 70)
        print(f"Running moons experiment with hidden_dim = {hidden_dim}")
        print("=" * 70)

        model, history, selected_epoch = train_nn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2,
            optimizer="adam",
            lr=0.01,
            reg_lambda=0,
            batch_size=64,
            epochs=1000,
            seed=42,
            checkpoint_policy="final",   # synthetic tasks üçün final epoch normaldır
            verbose=True,
        )

        train_metrics = evaluate(model, X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)

        plot_decision_boundary(
            model=model,
            X=X_all,
            y=y_all,
            title=f"Moons Decision Boundary (hidden={hidden_dim})",
            save_path=figures_dir / f"moons_boundary_{hidden_dim}.png",
        )

        plot_training_curves(
            history=history,
            title=f"Moons Training Curves (hidden={hidden_dim})",
            save_path=figures_dir / f"moons_curves_{hidden_dim}.png",
        )

        row = {
            "hidden_dim": hidden_dim,
            "selected_epoch": int(selected_epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        summary.append(row)

        print(f"\nHidden dim: {hidden_dim}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['accuracy']:.4f}")
        print(f"Test  Loss: {test_metrics['loss']:.4f} | Test  Acc: {test_metrics['accuracy']:.4f}\n")

    out_path = results_dir / "moons_capacity_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for row in summary:
        print(
            f"hidden={row['hidden_dim']:>2} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"test_acc={row['test_accuracy']:.4f} | "
            f"test_loss={row['test_loss']:.4f}"
        )

    print(f"\nSaved results to: {out_path}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()