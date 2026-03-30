"""
Math4AI Capstone: Evaluation Metrics
National AI Center — Academy

This module implements the core quantitative measures used to assess model 
performance, following the official capstone protocol (§4.2).

Metrics provided:
    - Accuracy: Overall classification success rate.
    - Cross-Entropy: Primary probabilistic loss for model selection.
    - Confusion Matrix: Error analysis across classes.
    - Reliability Table: Binning for Track B (Uncertainty analysis).
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard classification accuracy.
    Calculates the ratio of correctly predicted labels to total samples.
    """
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def mean_cross_entropy(P: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculates the average Negative Log-Likelihood (NLL).
    
    Formula: L = -(1/n) * Σ [y_i * log(p_i)]
    Used as the primary metric for model checkpointing and selection.
    """
    n = Y.shape[0]
    # Numerical stability: clip probabilities to avoid log(0)
    eps = 1e-15
    P_safe = np.clip(P, eps, 1.0)
    
    return float(-np.sum(Y * np.log(P_safe)) / n)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """
    Constructs an error distribution matrix.
    
    Args:
        y_true: Ground truth integer labels.
        y_pred: Model predictions.
        k: Number of classes.
        
    Returns:
        C: (k x k) matrix where C[i, j] is the count of class i predicted as j.
    """
    C = np.zeros((k, k), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label < k and pred_label < k:
            C[true_label, pred_label] += 1
    return C


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """
    Computes accuracy individually for each class.
    Essential for identifying which specific digits (e.g., 1 vs 7) the model confuses.
    """
    class_accs = np.zeros(k)
    for c in range(k):
        mask = (y_true == c)
        if np.any(mask):
            class_accs[c] = np.mean(y_pred[mask] == c)
        else:
            class_accs[c] = 0.0
    return class_accs


def confidence_reliability_table(confidences: np.ndarray, 
                                 is_correct: np.ndarray, 
                                 n_bins: int = 5) -> np.ndarray:
    """
    Track B: Reliability analysis (Calibration).
    
    Groups predictions into bins based on the model's confidence levels 
    and compares average confidence to actual empirical accuracy.
    
    Returns:
        A (n_bins x 4) table with:
        [bin_min, bin_max, mean_confidence_in_bin, accuracy_in_bin]
    """
    # Create bin boundaries (e.g., 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    stats_table = []

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i+1]
        
        # Select samples that fall into this confidence interval
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
            
        if np.any(mask):
            bin_acc = np.mean(is_correct[mask])
            bin_conf = np.mean(confidences[mask])
            stats_table.append([lower, upper, bin_conf, bin_acc])
        else:
            # If no samples fall in this bin, use NaN to avoid misleading zeros
            stats_table.append([lower, upper, np.nan, np.nan])

    return np.array(stats_table)