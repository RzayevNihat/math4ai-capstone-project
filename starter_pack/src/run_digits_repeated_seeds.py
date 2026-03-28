import json
from pathlib import Path

import numpy as np

from train_nn import train_nn


def load_digits():
    data = np.load("starter_pack/data/digits_data.npz")
    split = np.load("starter_pack/data/digits_split_indices.npz")

    X = data["X"]
    y = data["y"]

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    X_train = np.asarray(X[train_idx], dtype=np.float64)
    y_train = np.asarray(y[train_idx], dtype=np.int64)

    X_val = np.asarray(X[val_idx], dtype=np.float64)
    y_val = np.asarray(y[val_idx], dtype=np.int64)

    X_test = np.asarray(X[test_idx], dtype=np.float64)
    y_test = np.asarray(y[test_idx], dtype=np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model, X, y):
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


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    half_width = 2.776 * std / np.sqrt(len(values))
    return float(mean), float(half_width)


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

    # Final selected NN config
    # Based on your optimizer study, you can keep adam or change later.
    final_config = {
        "hidden_dim": 32,
        "optimizer": "adam",
        "lr": 0.001,
        "reg_lambda": 1e-4,
        "batch_size": 64,
        "epochs": 200,
        "checkpoint_policy": "best_val",
    }

    seeds = [11, 22, 33, 44, 55]

    run_results = []

    for seed in seeds:
        print("=" * 70)
        print(f"Running seed = {seed}")
        print("=" * 70)

        model, history, best_epoch = train_nn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=X_train.shape[1],
            hidden_dim=final_config["hidden_dim"],
            output_dim=len(np.unique(y_train)),
            optimizer=final_config["optimizer"],
            lr=final_config["lr"],
            reg_lambda=final_config["reg_lambda"],
            batch_size=final_config["batch_size"],
            epochs=final_config["epochs"],
            seed=seed,
            checkpoint_policy=final_config["checkpoint_policy"],
            verbose=False,
        )

        test_metrics = evaluate_model(model, X_test, y_test)

        row = {
            "seed": seed,
            "best_epoch": int(best_epoch),
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
        }
        run_results.append(row)

        print(
            f"Seed {seed} | "
            f"Best Epoch: {best_epoch} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f}"
        )

    accs = [r["test_accuracy"] for r in run_results]
    losses = [r["test_loss"] for r in run_results]

    mean_acc, ci_acc = ci95(accs)
    mean_loss, ci_loss = ci95(losses)

    summary = {
        "final_config": final_config,
        "runs": run_results,
        "mean_test_accuracy": mean_acc,
        "ci95_test_accuracy": ci_acc,
        "mean_test_loss": mean_loss,
        "ci95_test_loss": ci_loss,
    }

    results_dir = Path("starter_pack/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "digits_repeated_seeds_nn.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Mean Test Accuracy : {mean_acc:.4f} ± {ci_acc:.4f}")
    print(f"Mean Test Loss     : {mean_loss:.4f} ± {ci_loss:.4f}")
    print(f"Saved results to   : {out_path}")


if __name__ == "__main__":
    main()