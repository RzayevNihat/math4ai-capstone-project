"""
Math4AI Capstone: Data Utilities
National AI Center — Academy

This module provides essential preprocessing functions for the project:
    - One-hot encoding for multiclass labels.
    - Standard loading procedures for the Digits benchmark.
    - Normalization utilities for synthetic datasets.
"""

import numpy as np


def one_hot_encode(y: np.ndarray, k: int) -> np.ndarray:
    """
    Convert integer class labels into a one-hot encoded matrix.
    
    Args:
        y (np.ndarray): Array of class indices (shape: n).
        k (int): Total number of target classes.
        
    Returns:
        Y (np.ndarray): One-hot matrix (shape: n x k).
    """
    n = y.shape[0]
    Y = np.zeros((n, k))
    Y[np.arange(n), y] = 1.0
    return Y


def load_digits(data_path: str, split_path: str) -> dict:
    """
    Load the Digits benchmark using fixed index splits.
    
    Per Protocol (§4.2):
    - Features are already scaled [0, 1]; additional scaling is skipped.
    - Uses predefined 'train_idx', 'val_idx', and 'test_idx' to ensure
      reproducibility across all student models.
    """
    # Load raw data and the split indices
    raw_data = np.load(data_path)
    split_indices = np.load(split_path)

    X, y = raw_data["X"], raw_data["y"]

    # Flatten image tensors if necessary (e.g., from (n, 8, 8) to (n, 64))
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    # Apply the capstone-mandated splits
    tr_idx = split_indices["train_idx"]
    vl_idx = split_indices["val_idx"]
    ts_idx = split_indices["test_idx"]

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[vl_idx], y[vl_idx]
    X_test,  y_test  = X[ts_idx], y[ts_idx]

    # Detect class count and generate one-hot labels
    k = int(len(np.unique(y)))
    Y_train = one_hot_encode(y_train, k)
    Y_val   = one_hot_encode(y_val, k)
    Y_test  = one_hot_encode(y_test, k)

    return {
        "X_train": X_train, "Y_train": Y_train, "y_train": y_train,
        "X_val":   X_val,   "Y_val":   Y_val,   "y_val":   y_val,
        "X_test":  X_test,  "Y_test":  Y_test,  "y_test":  y_test,
        "n_features": X_train.shape[1],
        "n_classes": k
    }


def load_synthetic(data_path: str) -> dict:
    """
    Load non-benchmark synthetic datasets (e.g., Moons or Gaussian clusters).
    Handles both pre-split and raw formats.
    """
    data = np.load(data_path)
     # Handle both (X, y) and pre-split formats
    if "X_train" in data:
        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
        # Create a small validation set from test if not present
        split_point = len(X_test) // 2
        X_val = data.get("X_val", X_test[:split_point])
        y_val = data.get("y_val", y_test[:split_point])
    else:
        # Manual 60/20/20 split for raw synthetic data
        X, y = data["X"], data["y"]
        n = len(y)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        
        m1, m2 = int(0.6 * n), int(0.8 * n)
        X_train, y_train = X[idx[:m1]],    y[idx[:m1]]
        X_val,   y_val   = X[idx[m1:m2]], y[idx[m1:m2]]
        X_test,  y_test  = X[idx[m2:]],    y[idx[m2:]]

    k = int(len(np.unique(y_train)))
    return {
        "X_train": X_train, "Y_train": one_hot_encode(y_train, k), "y_train": y_train,
        "X_val":   X_val,   "Y_val":   one_hot_encode(y_val, k),   "y_val":   y_val,
        "X_test":  X_test,  "Y_test":  one_hot_encode(y_test, k),  "y_test":  y_test,
        "n_features": X_train.shape[1],
        "n_classes": k
    }


def standardize(X_train: np.ndarray, 
                X_val: np.ndarray = None, 
                X_test: np.ndarray = None) -> tuple:
    """
    Perform Z-score normalization (zero mean, unit variance).
    Statistics are computed ONLY on the training set to prevent data leakage.
    
    Warning: Do not apply to the Digits dataset.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-9 # Small epsilon to prevent division by zero

    def _apply(data):
        return (data - mu) / sigma if data is not None else None

    return _apply(X_train), _apply(X_val), _apply(X_test)