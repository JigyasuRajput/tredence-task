"""Mini-batched training loop and an accuracy helper."""

import numpy as np

from engine import Tensor, softmax_cross_entropy


def accuracy(model, X, y):
    """Fraction of correct predictions; argmax of the logits needs no softmax."""
    preds = model(Tensor(X)).data.argmax(axis=1)
    return float((preds == y).mean())


def train(model, optimizer, X_train, y_train, X_test, y_test,
          epochs=30, batch_size=64, seed=0):
    """Train for some epochs, recording per-epoch loss and train/test accuracy."""
    rng = np.random.default_rng(seed)
    n = len(X_train)
    history = {"loss": [], "train_acc": [], "test_acc": []}

    for _ in range(epochs):
        perm = rng.permutation(n)
        running_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = softmax_cross_entropy(model(Tensor(X_train[idx])), y_train[idx])
            loss.backward()          # backward() re-zeros grads, so no stale accumulation
            optimizer.step()
            running_loss += float(loss.data)
            n_batches += 1
        history["loss"].append(running_loss / n_batches)
        history["train_acc"].append(accuracy(model, X_train, y_train))
        history["test_acc"].append(accuracy(model, X_test, y_test))

    return history
