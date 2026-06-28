"""Train while gradually pruning the network to a target sparsity."""

import numpy as np

from engine import Tensor, softmax_cross_entropy
from prune import cubic_sparsity, saliency

from .loop import accuracy


def _full_batch_grads(model, X, y):
    """One forward+backward over the whole set, so saliency uses a stable gradient
    rather than a single noisy minibatch."""
    softmax_cross_entropy(model(Tensor(X)), y).backward()


def train_pruned(model, optimizer, pruner, X_train, y_train, X_test, y_test,
                 final_sparsity=0.9, importance=saliency, epochs=40,
                 batch_size=64, prune_fraction=0.7, seed=0):
    """Train, gradually pruning to final_sparsity over the first part of training.

    Pruning happens at the end of each epoch in the pruning window, following the
    cubic ramp; the remaining epochs let the network recover at full sparsity.
    """
    rng = np.random.default_rng(seed)
    n = len(X_train)
    prune_window = max(1, int(epochs * prune_fraction))
    history = {"loss": [], "test_acc": [], "sparsity": []}

    for epoch in range(epochs):
        perm = rng.permutation(n)
        running, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = softmax_cross_entropy(model(Tensor(X_train[idx])), y_train[idx])
            loss.backward()
            optimizer.step()
            running += float(loss.data)
            n_batches += 1

        if epoch < prune_window:
            target = cubic_sparsity(epoch + 1, prune_window, final_sparsity)
            _full_batch_grads(model, X_train, y_train)
            pruner.prune_to(target, importance)

        history["loss"].append(running / n_batches)
        history["test_acc"].append(accuracy(model, X_test, y_test))
        history["sparsity"].append(pruner.sparsity())

    return history
