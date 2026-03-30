"""
Math4AI Capstone: Softmax Regression (Model 1)
National AI Center — Academy

This module implements a linear multiclass classifier using the Softmax function
and Cross-Entropy loss. It follows the project notation (§2.2):
    X : input batch (n, d)
    W : weights (k, d)
    b : bias (1, k)
    Y : one-hot labels (n, k)
"""

import numpy as np

# ============================================================
# Core Mathematical Functions
# ============================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of each row in the input matrix.
    Uses the max-subtraction trick to ensure numerical stability.
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)): 
        raise RuntimeError("Numerical instability detected in softmax (NaN/Inf).")
    return probs


def cross_entropy_loss(probs: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the average negative log-likelihood for a batch.
    Includes a small epsilon clip to prevent log(0).
    """
    n = Y.shape[0]
    safe_probs = np.clip(probs, 1e-15, 1.0)
    return -np.sum(Y * np.log(safe_probs)) / n


# ============================================================
# Optimizer Helpers
# ============================================================

def _make_opt_state(params: list, opt_name: str) -> list[dict]:
    """
    Initialize internal buffers (velocity, moments) for the chosen optimizer.
    """
    states = []
    for p in params:
        if opt_name == "momentum":
            states.append({"vel": np.zeros_like(p)})
        elif opt_name == "adam":
            states.append({
                "first_moment": np.zeros_like(p),
                "second_moment": np.zeros_like(p),
                "step": 0
            })
        else:
            states.append({}) # Vanilla SGD
    return states


def _step(param: np.ndarray, grad: np.ndarray, state: dict, 
          opt_name: str, lr: float, mu: float = 0.9, 
          beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
    """
    Perform a single update step for a parameter using the specified optimizer.
    Includes gradient clipping to stabilize training.
    """
    grad = np.clip(grad, -5.0, 5.0)

    if opt_name == "sgd":
        param -= lr * grad

    elif opt_name == "momentum":
        vel = state["vel"]
        vel[:] = mu * vel + lr * grad
        param -= vel

    elif opt_name == "adam":
        state["step"] += 1
        t, m, v = state["step"], state["first_moment"], state["second_moment"]
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    else: 
        raise ValueError(f"Unknown optimizer: {opt_name}")

# ============================================================
# SoftmaxRegression Class
# ============================================================

class SoftmaxRegression:
    """
    Linear multiclass classifier trained with Softmax and Cross-Entropy.
    Supports SGD, Momentum, and Adam optimizers.
    """

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.05, 
                  reg: float = 1e-4, optimizer: str = "sgd", seed: int = 42):
        """Initialize weights using Xavier-style normalization."""
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, np.sqrt(1.0 / n_features), size=(n_classes, n_features))
        self.b = np.zeros((1, n_classes))

        self.lr, self.reg = lr, reg
        self.optimizer = optimizer.lower()
        self._opt_state = None

        # Training history
        self.train_loss_hist, self.val_loss_hist = [], []
        self.train_acc_hist, self.val_acc_hist = [], []

        # Checkpointing
        self.best_val_loss = np.inf
        self.best_W, self.best_b = self.W.copy(), self.b.copy()
        self.best_epoch = 0

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities: probs = softmax(XW^T + b)."""
        self._cache_X = X
        self._scores = X @ self.W.T + self.b
        self._probs = softmax(self._scores)
        return self._probs

    def backward_pass(self, Y: np.ndarray) -> None:
        """Compute gradients and update weights using the optimizer."""
        n = Y.shape[0]
        if self._opt_state is None: 
            self._opt_state = _make_opt_state([self.W, self.b], self.optimizer)
        
        # Gradient of CE w.r.t logits
        dS = (self._probs - Y) / n
        
        # Gradients w.r.t parameters (with L2 regularization for W)
        dW = dS.T @ self._cache_X + self.reg * self.W
        db = np.sum(dS, axis=0, keepdims=True)

        _step(self.W, dW, self._opt_state[0], self.optimizer, self.lr)
        _step(self.b, db, self._opt_state[1], self.optimizer, self.lr)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
              X_val: np.ndarray = None, Y_val: np.ndarray = None, 
              epochs: int = 200, batch_size: int = 64, verbose: bool = True):
        """Main training loop with mini-batch SGD and validation monitoring."""
        n = X_train.shape[0]
        has_val = (X_val is not None) and (Y_val is not None)

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n)
            X_shuf, Y_shuf = X_train[perm], Y_train[perm]
            for start in range(0, n, batch_size):
                Xb, Yb = X_shuf[start : start + batch_size], Y_shuf[start : start + batch_size]
                self.forward_pass(Xb)
                self.backward_pass(Yb)

            # Epoch-end metrics
            p_tr = self.forward_pass(X_train)
            tr_loss = cross_entropy_loss(p_tr, Y_train) + 0.5 * self.reg * np.sum(self.W**2)
            tr_acc = self._acc(p_tr, Y_train)
            
            self.train_loss_hist.append(tr_loss)
            self.train_acc_hist.append(tr_acc)

            if has_val:
                p_vl = self.forward_pass(X_val)
                vl_loss = cross_entropy_loss(p_vl, Y_val)
                vl_acc = self._acc(p_vl, Y_val)
                self.val_loss_hist.append(vl_loss)
                self.val_acc_hist.append(vl_acc)
                if vl_loss < self.best_val_loss:
                    self.best_val_loss, self.best_epoch = vl_loss, epoch
                    self.best_W, self.best_b = self.W.copy(), self.b.copy()

            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"Epoch {epoch:3}: Loss={tr_loss:.4f} | Acc={tr_acc:.4f}")

    def load_best_weights(self):
        """Restore model to the state with the best validation performance."""
        self.W, self.b = self.best_W.copy(), self.best_b.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class labels."""
        return np.argmax(self.forward_pass(X), axis=1)

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Return classification accuracy on the given dataset."""
        return self._acc(self.forward_pass(X), Y)

    def mean_cross_entropy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Return average CE loss (no regularization)."""
        return cross_entropy_loss(self.forward_pass(X), Y)

    # --- Track B Methods ---

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Track B: Returns the maximum predicted probability for each sample."""
        probs = self.forward_pass(X)
        return np.max(probs, axis=1)

    def predict_entropy(self, X: np.ndarray) -> np.ndarray:
        """Track B: Returns the predictive entropy for each sample."""
        probs = np.clip(self.forward_pass(X), 1e-15, 1.0)
        return -np.sum(probs * np.log(probs), axis=1)

    @staticmethod
    def _acc(probs, Y):
        return float(np.mean(np.argmax(probs, axis=1) == np.argmax(Y, axis=1)))


# ============================================================
# Statistical Evaluation
# ============================================================

def repeated_seed_evaluation(X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                             n_features, n_classes, epochs=200, lr=0.05, 
                             reg=1e-4, seeds=(0,1,2,3,4)):
    """
    Evaluate the model over multiple seeds to compute mean and 95% CI.
    Helps ensure that results are not due to initialization luck.
    """
    accs, ces = [], []
    for s in seeds:
        # Düzəliş: reg parametri buraya əlavə edildi
        model = SoftmaxRegression(n_features, n_classes, lr=lr, reg=reg, seed=s)
        model.train(X_train, Y_train, X_val, Y_val, epochs=epochs, verbose=False)
        model.load_best_weights()
        accs.append(model.accuracy(X_test, Y_test))
        ces.append(model.mean_cross_entropy(X_test, Y_test))

    accs, ces = np.array(accs), np.array(ces)
    return {
        "mean_acc": accs.mean(),
        "ci_acc": 2.776 * accs.std(ddof=1) / np.sqrt(len(seeds)),
        "mean_ce": ces.mean(),
        "ci_ce": 2.776 * ces.std(ddof=1) / np.sqrt(len(seeds))
    }