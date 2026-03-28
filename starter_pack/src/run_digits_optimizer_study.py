import json
import numpy as np
from pathlib import Path

from train_nn import train_nn


def load_digits_data():
    """
    Load digits dataset and fixed train/val/test split.
    Assumes this script is run from the project root or the file paths are valid.
    """
    data = np.load("starter_pack/data/digits_data.npz")
    split = np.load("starter_pack/data/digits_split_indices.npz")

    X = data["X"]
    y = data["y"]

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test, X.shape[1], len(np.unique(y))


def evaluate(model, X, y):
    """
    Evaluate a trained model on a dataset.
    """
    probs = model.forward(X)
    loss = model.compute_loss(y)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)

    return {
        "loss": float(loss),
        "accuracy": float(acc),
    }


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, input_dim, output_dim = load_digits_data()

    # Assignment-required optimizer settings
    configs = [
        {
            "name": "sgd",
            "optimizer": "sgd",
            "lr": 0.05,
            "momentum_beta": 0.9,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        {
            "name": "momentum",
            "optimizer": "momentum",
            "lr": 0.05,
            "momentum_beta": 0.9,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        {
            "name": "adam",
            "optimizer": "adam",
            "lr": 0.001,
            "momentum_beta": 0.9,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
    ]

    results = []

    for cfg in configs:
        print("=" * 70)
        print(f"Running optimizer: {cfg['name']}")
        print("=" * 70)

        model, history, best_epoch = train_nn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            hidden_dim=32,              # required default
            output_dim=output_dim,
            optimizer=cfg["optimizer"],
            lr=cfg["lr"],
            momentum_beta=cfg["momentum_beta"],
            beta1=cfg["beta1"],
            beta2=cfg["beta2"],
            eps=cfg["eps"],
            reg_lambda=1e-4,            # required default
            batch_size=64,              # required default
            epochs=200,                 # required budget
            seed=42,
            checkpoint_policy="best_val",
            verbose=True,
        )

        train_metrics = evaluate(model, X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)

        result = {
            "optimizer": cfg["name"],
            "best_epoch": int(best_epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "history": {
                "train_loss": history["train_loss"],
                "train_acc": history["train_acc"],
                "val_loss": history["val_loss"],
                "val_acc": history["val_acc"],
            },
        }

        results.append(result)

        print(f"\nOptimizer: {cfg['name']}")
        print(f"Best epoch: {best_epoch}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['accuracy']:.4f}")
        print(f"Test  Loss: {test_metrics['loss']:.4f} | Test  Acc: {test_metrics['accuracy']:.4f}")
        print()

    # Save results
    results_dir = Path("starter_pack/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "digits_optimizer_study.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(
            f"{r['optimizer']:>10} | "
            f"best_epoch={r['best_epoch']:>3} | "
            f"val_loss={r['val_loss']:.4f} | "
            f"val_acc={r['val_accuracy']:.4f} | "
            f"test_loss={r['test_loss']:.4f} | "
            f"test_acc={r['test_accuracy']:.4f}"
        )

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()