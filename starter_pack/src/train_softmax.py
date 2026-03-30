"""
Math4AI Capstone: Softmax Regression Training Pipeline
National AI Center — AI Academy

This script executes the full training and evaluation protocol for Model 1:
    1. Data Loading: Digits benchmark (8x8 images).
    2. Baseline Training: Standard SGD with L2 regularization.
    3. Optimizer Study: Performance comparison (SGD vs. Momentum vs. Adam).
    4. Stability Analysis: 95% Confidence Intervals via repeated seeds.

Results are saved to the 'results/' directory for subsequent plotting.
"""

import os
import sys
import numpy as np

# Ensure the project root is in the path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.softmax_regression import SoftmaxRegression, repeated_seed_evaluation
from utils.data_utils import load_digits
from utils.metrics import accuracy
# ============================================================
# Global Configuration & Paths
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'digits_data.npz')
SPLIT_PATH = os.path.join(BASE_DIR, 'data', 'digits_split_indices.npz')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Main Execution Logic
# ============================================================

def main() -> None:
    """
    Runs the complete experiment suite for Softmax Regression.
    """
    np.random.seed(42)

    # --- 1. Data Loading ---
    print(f"Loading digits benchmark from: {DATA_PATH}")
    try:
        dataset = load_digits(DATA_PATH, SPLIT_PATH)
    except FileNotFoundError as e:
        print(f"Error: Required data files not found. {e}")
        sys.exit(1)

    X_train, Y_train = dataset['X_train'], dataset['Y_train']
    X_val,   Y_val   = dataset['X_val'],   dataset['Y_val']
    X_test,  Y_test  = dataset['X_test'],  dataset['Y_test']
    y_test           = dataset['y_test']
    
    n_features, n_classes = dataset['n_features'], dataset['n_classes']

    print(f"  Dataset: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"  Dimensions: Features={n_features}, Classes={n_classes}")

    # --- 2. Baseline Model Training (Protocol §4.2) ---
    # Parameters are fixed per capstone requirements to ensure fair comparison
    lr, reg_const = 0.05, 1e-4
    batch_size, epochs = 64, 200

    print("\n[Baseline] Training Softmax Regression (SGD)...")
    model = SoftmaxRegression(n_features, n_classes, lr=lr, reg=reg_const, optimizer='sgd')
    
    model.train(X_train, Y_train, X_val, Y_val, 
                epochs=epochs, batch_size=batch_size, verbose=True)

    # Roll back to the best validation checkpoint before final evaluation
    model.load_best_weights()

    print("\nTest Set Evaluation:")
    test_preds = model.predict(X_test)
    test_acc = accuracy(y_test, test_preds)
    test_ce = model.mean_cross_entropy(X_test, Y_test)
    
    print(f"  Accuracy      : {test_acc:.4f}")
    print(f"  Cross-Entropy : {test_ce:.4f}")
    print(f"  Best Epoch    : {model.best_epoch}")

    # Save training history for learning curve visualization
    history_path = os.path.join(RESULTS_DIR, 'softmax_training_history.npz')
    np.savez(history_path, 
             loss_history=np.array(model.train_loss_hist),
             val_loss_history=np.array(model.val_loss_hist),
             train_acc_history=np.array(model.train_acc_hist),
             val_acc_history=np.array(model.val_acc_hist))

    # --- 3. Optimizer Comparison Study ---
    # Evaluate how different update rules affect the convergence of a linear model
    print("\n[Study] Comparing Optimizers (SGD vs. Momentum vs. Adam)...")
    opt_configs = [('sgd', 0.05), ('momentum', 0.05), ('adam', 0.001)]
    opt_results = {}

    for name, opt_lr in opt_configs:
        m = SoftmaxRegression(n_features, n_classes, lr=opt_lr, reg=reg_const, optimizer=name, seed=0)
        m.train(X_train, Y_train, X_val, Y_val, epochs=epochs, batch_size=batch_size, verbose=False)
        m.load_best_weights()
        
        opt_results[name] = {
            'acc': m.accuracy(X_test, Y_test),
            'ce': m.mean_cross_entropy(X_test, Y_test),
            'train_loss': np.array(m.train_loss_hist),
            'val_loss': np.array(m.val_loss_hist)
        }
        print(f"  {name:10s} | Acc: {opt_results[name]['acc']:.4f} | CE: {opt_results[name]['ce']:.4f}")

    # Save optimizer study results
    opt_path = os.path.join(RESULTS_DIR, 'softmax_optimizer_study.npz')
    np.savez(opt_path, **{f"{opt}_{k}": v for opt, res in opt_results.items() for k, v in res.items()})

    # --- 4. Repeated Seed Evaluation ---
    # Statistical validation using 5 different initializations
    print("\n[Statistical] Running 5-seed repeated evaluation...")
    seed_stats = repeated_seed_evaluation(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        n_features, n_classes, epochs=epochs, lr=lr, reg=reg_const
    )

    rs_path = os.path.join(RESULTS_DIR, 'softmax_repeated_seeds.npz')
    np.savez(rs_path, **seed_stats)
    
    print(f"  Mean Accuracy : {seed_stats['mean_acc']:.4f} +/- {seed_stats['ci_acc']:.4f} (95% CI)")
    print(f"\nDone. All results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
