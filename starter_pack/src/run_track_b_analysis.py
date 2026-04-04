import json
from pathlib import Path

import numpy as np

from train_nn import train_nn
from models.softmax_regression import SoftmaxRegression


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


def predictive_entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def reliability_bins(confidence, correct, n_bins=5):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_index": i,
                    "bin_start": float(lo),
                    "bin_end": float(hi),
                    "count": 0,
                    "mean_confidence": None,
                    "empirical_accuracy": None,
                }
            )
            continue

        rows.append(
            {
                "bin_index": i,
                "bin_start": float(lo),
                "bin_end": float(hi),
                "count": count,
                "mean_confidence": float(np.mean(confidence[mask])),
                "empirical_accuracy": float(np.mean(correct[mask])),
            }
        )

    return rows


def analyze_predictions(probs, y_true):
    preds = np.argmax(probs, axis=1)
    correct_mask = preds == y_true

    conf = np.max(probs, axis=1)
    ent = predictive_entropy(probs)

    result = {
        "accuracy": float(np.mean(correct_mask)),
        "mean_confidence": float(np.mean(conf)),
        "mean_entropy": float(np.mean(ent)),
        "reliability_bins": reliability_bins(conf, correct_mask.astype(np.float64), n_bins=5),
        "correct_vs_incorrect": {
            "confidence_correct_mean": float(np.mean(conf[correct_mask])) if np.any(correct_mask) else None,
            "confidence_incorrect_mean": float(np.mean(conf[~correct_mask])) if np.any(~correct_mask) else None,
            "entropy_correct_mean": float(np.mean(ent[correct_mask])) if np.any(correct_mask) else None,
            "entropy_incorrect_mean": float(np.mean(ent[~correct_mask])) if np.any(~correct_mask) else None,
        },
        "raw": {
            "confidence": conf.tolist(),
            "entropy": ent.tolist(),
            "correct_mask": correct_mask.astype(int).tolist(),
            "preds": preds.tolist(),
            "y_true": y_true.tolist(),
        },
    }
    return result


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    Y_train = one_hot(y_train, n_classes)
    Y_val = one_hot(y_val, n_classes)

    # ---------------------------
    # Softmax
    # ---------------------------
    softmax_model = SoftmaxRegression(
        n_features=n_features,
        n_classes=n_classes,
        lr=0.05,
        l2_lambda=1e-4,
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
    softmax_model.restore_best_checkpoint()
    softmax_probs = softmax_model.predict_proba(X_test)

    # ---------------------------
    # NN
    # ---------------------------
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
    nn_probs = nn_model.forward(X_test)

    summary = {
        "softmax": analyze_predictions(softmax_probs, y_test),
        "nn": analyze_predictions(nn_probs, y_test),
    }

    out_dir = Path("starter_pack/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "track_b_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("TRACK B ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()