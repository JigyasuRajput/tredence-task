"""Globally rank connections by importance and enforce a hard sparsity budget."""

import numpy as np


def prunable_weights(model):
    """The 2-D weight matrices we prune; biases (1-D) are left dense."""
    return [p for p in model.parameters() if p.data.ndim == 2]


class Pruner:
    """Holds the weight matrices and masks off the least-important connections."""

    def __init__(self, weights):
        self.weights = list(weights)

    def total(self):
        return int(sum(w.size for w in self.weights))

    def n_pruned(self):
        return int(sum(int((w.mask == 0.0).sum()) for w in self.weights))

    def sparsity(self):
        return self.n_pruned() / self.total()

    def prune_to(self, sparsity, importance):
        """Mask exactly `sparsity` fraction of all connections by lowest importance.

        Ranking is global across layers, so a tiny weight in one layer is dropped
        before a large weight in another. Already-dead weights score zero, so they
        stay dead as the target rises — the schedule only ever removes more.
        """
        scores = [importance(w) for w in self.weights]
        flat = np.concatenate([s.ravel() for s in scores])
        n_total = flat.size
        n_prune = min(max(int(round(sparsity * n_total)), 0), n_total)

        keep = np.ones(n_total, dtype=bool)
        if n_prune > 0:
            # Indices of the n_prune smallest scores — an exact count, so the budget
            # is hit precisely rather than drifting on threshold ties.
            prune_idx = np.argpartition(flat, n_prune - 1)[:n_prune]
            keep[prune_idx] = False

        offset = 0
        for w, s in zip(self.weights, scores):
            n = s.size
            w.mask = keep[offset:offset + n].reshape(s.shape).astype(np.float64)
            w.apply_mask()
            offset += n
