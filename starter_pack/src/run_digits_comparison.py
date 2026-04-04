import json
import sys
from pathlib import Path

import numpy as np

# make sibling imports robust
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from train_nn import train_nn
from softmax_regression import SoftmaxRegression


def one_hot(y, num_classes):
    y = np.asarray(y, dtype=np.int64)
    Y = np.zeros((len(y), num_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def load_digits():
    data = np.load("starter_pack/data/digits_data.npz")
    split = np.load("starter_pack/data/digits_split_indices.npz")

    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64)

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_nn(model, X, y):
    probs = model.forward(X)
    loss = model.compute_loss(y)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "preds": preds.tolist(),
    }


def evaluate_softmax(model, X, Y_onehot):
    probs = model.forward_pass(X)
    loss = model.mean_cross_entropy(X, Y_onehot)
    preds = np.argmax(probs, axis=1)
    acc = model.accuracy(X, Y_onehot)

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "preds": preds.tolist(),
    }


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    Y_train = one_hot(y_train, n_classes)
    Y_val = one_hot(y_val, n_classes)
    Y_test = one_hot(y_test, n_classes)

    print("=" * 70)
    print("DIGITS COMPARISON: SOFTMAX vs NN")
    print("=" * 70)

    # -------------------------------------------------
    # Softmax baseline
    # -------------------------------------------------
    softmax_model = SoftmaxRegression(
        n_features=n_features,
        n_classes=n_classes,
        lr=0.05,
        reg=1e-4,
        optimizer="sgd",
        seed=42,
    )

    softmax_model.train(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        epochs=200,
        batch_size=64,
        verbose=False,
    )
    softmax_model.load_best_weights()

    softmax_test = evaluate_softmax(softmax_model, X_test, Y_test)

    print(
        f"Softmax | Best Epoch: {softmax_model.best_epoch} | "
        f"Test Acc: {softmax_test['accuracy']:.4f} | "
        f"Test Loss: {softmax_test['loss']:.4f}"
    )

    # -------------------------------------------------
    # Final NN config
    # -------------------------------------------------
    nn_model, nn_history, nn_best_epoch = train_nn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=n_features,
        hidden_dim=32,
        output_dim=n_classes,
        optimizer="adam",
        lr=0.001,
        reg_lambda=1e-4,
        batch_size=64,
        epochs=200,
        seed=42,
        checkpoint_policy="best_val",
        verbose=False,
    )

    nn_test = evaluate_nn(nn_model, X_test, y_test)

    print(
        f"NN      | Best Epoch: {nn_best_epoch} | "
        f"Test Acc: {nn_test['accuracy']:.4f} | "
        f"Test Loss: {nn_test['loss']:.4f}"
    )

    results = {
        "softmax": {
            "best_epoch": int(softmax_model.best_epoch),
            "test_accuracy": softmax_test["accuracy"],
            "test_loss": softmax_test["loss"],
        },
        "nn": {
            "best_epoch": int(nn_best_epoch),
            "test_accuracy": nn_test["accuracy"],
            "test_loss": nn_test["loss"],
        },
    }

    out_dir = Path("starter_pack/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "digits_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()