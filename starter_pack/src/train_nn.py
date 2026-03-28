import numpy as np
from nn_model import OneHiddenLayerNN


def iterate_minibatches(X, y, batch_size=64, shuffle=True, seed=None):
    """
    Split dataset into mini-batches.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Input features
    y : np.ndarray of shape (N,)
        Integer labels
    batch_size : int
        Number of samples in each mini-batch
    shuffle : bool
        Whether to shuffle before batching
    seed : int or None
        Random seed for reproducibility

    Yields
    ------
    X_batch, y_batch
    """
    N = X.shape[0]
    indices = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def evaluate(model, X, y):
    """
    Compute loss and accuracy on a dataset.
    """
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


def train_nn(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    hidden_dim,
    output_dim,
    optimizer="sgd",
    lr=None,
    momentum_beta=0.9,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    reg_lambda=1e-4,
    batch_size=64,
    epochs=200,
    seed=42,
    checkpoint_policy="best_val",
    verbose=True,
):
    
    # Default learning rates from assignment protocol
    if lr is None:
        if optimizer in ["sgd", "momentum"]:
            lr = 0.05
        elif optimizer == "adam":
            lr = 0.001
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    model = OneHiddenLayerNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        reg_lambda=reg_lambda,
        seed=seed,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state = model.state_dict()
    best_epoch = 0

    for epoch in range(epochs):
        # -------- TRAINING PHASE --------
        for X_batch, y_batch in iterate_minibatches(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            # forward pass
            model.forward(X_batch)

            # backward pass
            model.backward(y_batch)

            # parameter update
            model.update(
                optimizer=optimizer,
                lr=lr,
                momentum_beta=momentum_beta,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )

        # -------- EVALUATION PHASE --------
        train_eval = evaluate(model, X_train, y_train)
        val_eval = evaluate(model, X_val, y_val)

        history["train_loss"].append(train_eval["loss"])
        history["train_acc"].append(train_eval["accuracy"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["accuracy"])

        # Save best validation checkpoint
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_state = model.state_dict()
            best_epoch = epoch + 1

        if verbose:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"Train Loss: {train_eval['loss']:.4f} | "
                f"Train Acc: {train_eval['accuracy']:.4f} | "
                f"Val Loss: {val_eval['loss']:.4f} | "
                f"Val Acc: {val_eval['accuracy']:.4f}"
            )

    if checkpoint_policy == "best_val":
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    elif checkpoint_policy == "final":
        selected_epoch = epochs
    else:
        raise ValueError("checkpoint_policy must be 'best_val' or 'final'")

    return model, history, selected_epoch