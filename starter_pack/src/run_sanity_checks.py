import json
from pathlib import Path

import numpy as np

from nn_model import OneHiddenLayerNN
from train_nn import train_nn


def load_digits_small_subset():
    data = np.load("starter_pack/data/digits_data.npz")
    split = np.load("starter_pack/data/digits_split_indices.npz")

    X = data["X"]
    y = data["y"]

    train_idx = split["train_idx"]
    X_train = np.asarray(X[train_idx], dtype=np.float64)
    y_train = np.asarray(y[train_idx], dtype=np.int64)

    return X_train, y_train


def relative_error(a, b):
    return abs(a - b) / max(1e-12, abs(a) + abs(b))


def gradient_check_single_param(model, X, y, param_name, index, eps=1e-5):
    """
    Compare analytical gradient vs numerical gradient for one parameter entry.
    """
    # Analytical gradient
    model.forward(X)
    model.backward(y)
    analytic_grad = model.grads[param_name][index]

    # Numerical gradient
    param = getattr(model, param_name)
    original_value = param[index]

    param[index] = original_value + eps
    model.forward(X)
    loss_plus = model.compute_loss(y)

    param[index] = original_value - eps
    model.forward(X)
    loss_minus = model.compute_loss(y)

    param[index] = original_value  # restore

    numeric_grad = (loss_plus - loss_minus) / (2 * eps)
    rel_err = relative_error(analytic_grad, numeric_grad)

    return {
        "param": param_name,
        "index": tuple(int(i) for i in index),
        "analytic_grad": float(analytic_grad),
        "numeric_grad": float(numeric_grad),
        "relative_error": float(rel_err),
    }


def check_probabilities_sum_to_one(model, X):
    probs = model.forward(X)
    row_sums = np.sum(probs, axis=1)
    ok = np.allclose(row_sums, 1.0, atol=1e-7)
    return {
        "passed": bool(ok),
        "min_row_sum": float(np.min(row_sums)),
        "max_row_sum": float(np.max(row_sums)),
    }


def check_loss_decreases_on_tiny_subset(X_small, y_small):
    model = OneHiddenLayerNN(
        input_dim=X_small.shape[1],
        hidden_dim=16,
        output_dim=10,
        reg_lambda=1e-4,
        seed=42,
    )

    losses = []

    for _ in range(20):
        model.forward(X_small)
        losses.append(model.compute_loss(y_small))
        model.backward(y_small)
        model.update(optimizer="sgd",lr=0.1)

    return {
        "passed": bool(losses[-1] < losses[0]),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
    }


def check_tiny_subset_overfit(X_small, y_small):
    model, history, best_epoch = train_nn(
        X_train=X_small,
        y_train=y_small,
        X_val=X_small,
        y_val=y_small,
        input_dim=X_small.shape[1],
        hidden_dim=32,
        output_dim=10,
        optimizer="adam",
        lr=0.01,
        reg_lambda=0.0,
        batch_size=len(X_small),
        epochs=500,
        seed=123,
        checkpoint_policy="best_val",
        verbose=False,
    )

    probs = model.forward(X_small)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_small)

    return {
        "passed": bool(acc >= 0.95),
        "best_epoch": int(best_epoch),
        "tiny_subset_accuracy": float(acc),
    }


def check_nan_inf(model):
    arrays = [model.W1, model.b1, model.W2, model.b2]
    has_nan = any(np.isnan(arr).any() for arr in arrays)
    has_inf = any(np.isinf(arr).any() for arr in arrays)

    return {
        "passed": bool((not has_nan) and (not has_inf)),
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
    }


def main():
    X_train, y_train = load_digits_small_subset()

    # Use a tiny subset for sanity checks
    X_small = X_train[:12]
    y_small = y_train[:12]

    # For gradient checks, use a small model
    grad_model = OneHiddenLayerNN(
        input_dim=X_small.shape[1],
        hidden_dim=8,
        output_dim=10,
        reg_lambda=1e-4,
        seed=42,
    )

    gradient_checks = [
        gradient_check_single_param(grad_model, X_small, y_small, "W1", (0, 0)),
        gradient_check_single_param(grad_model, X_small, y_small, "W1", (3, 5)),
        gradient_check_single_param(grad_model, X_small, y_small, "W2", (0, 0)),
        gradient_check_single_param(grad_model, X_small, y_small, "W2", (2, 4)),
    ]

    prob_check = check_probabilities_sum_to_one(grad_model, X_small)
    loss_check = check_loss_decreases_on_tiny_subset(X_small, y_small)
    overfit_check = check_tiny_subset_overfit(X_small, y_small)
    nan_inf_check = check_nan_inf(grad_model)

    results = {
        "probabilities_sum_to_one": prob_check,
        "loss_decreases_on_tiny_subset": loss_check,
        "tiny_subset_overfit": overfit_check,
        "nan_inf_check": nan_inf_check,
        "gradient_checks": gradient_checks,
    }

    results_dir = Path("starter_pack/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "sanity_checks_nn.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("SANITY CHECKS SUMMARY")
    print("=" * 70)

    print("\n[1] Probability Sum Check")
    print(prob_check)

    print("\n[2] Loss Decrease on Tiny Subset")
    print(loss_check)

    print("\n[3] Tiny Subset Overfit Check")
    print(overfit_check)

    print("\n[4] NaN / Inf Check")
    print(nan_inf_check)

    print("\n[5] Gradient Checks")
    for item in gradient_checks:
        print(item)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()