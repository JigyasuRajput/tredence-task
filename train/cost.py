"""Honest cost measurement for a pruned model: active params, FLOPs, and a real sparse forward.

A dense matmul over zeroed weights is NOT cheaper, so the reproducible saving we report
is the drop in active parameters and the multiply-accumulate ops they imply. We also
provide a sparse-aware forward that touches only live connections (no multiply by zeros)
and matches the dense result. Honesty note: at this small scale the NumPy scatter that the
sparse path uses does fewer FLOPs but does NOT beat optimized dense BLAS in wall-clock — so
we claim a FLOP / active-parameter reduction, not a wall-clock speedup.
"""

import numpy as np

from nn import Linear, ReLU, Tanh
from prune import prunable_weights


def active_params(model):
    """Number of live (non-masked) weight connections."""
    return int(sum(int(w.mask.sum()) for w in prunable_weights(model)))


def total_params(model):
    """Number of weight connections if the model were dense."""
    return int(sum(w.size for w in prunable_weights(model)))


def sparse_linear(X, layer):
    """Forward through a Linear using only its live connections — no multiply by zeros."""
    rows, cols = np.nonzero(layer.weight.mask)        # active (in, out) index pairs
    vals = layer.weight.data[rows, cols]
    contrib = X[:, rows] * vals                       # one multiply per (sample, live edge)
    out = np.zeros((X.shape[0], layer.weight.shape[1]))
    np.add.at(out, (np.arange(X.shape[0])[:, None], cols), contrib)
    return out + layer.bias.data


def sparse_forward(model, X):
    """Run the model's forward with sparse Linear layers; matches the dense result."""
    h = np.asarray(X, dtype=np.float64)
    for layer in model.layers:
        if isinstance(layer, Linear):
            h = sparse_linear(h, layer)
        elif isinstance(layer, ReLU):
            h = np.maximum(h, 0.0)
        elif isinstance(layer, Tanh):
            h = np.tanh(h)
    return h


def matmul_macs(model, n_samples, only_active=False):
    """Multiply-accumulate ops for the weight matmuls over a batch of n_samples."""
    macs = 0
    for w in prunable_weights(model):
        connections = int(w.mask.sum()) if only_active else w.size
        macs += n_samples * connections
    return int(macs)


def measure_cost(model, n_samples):
    """Active params, sparsity, and the matmul FLOP reduction from skipping dead edges."""
    active = active_params(model)
    total = total_params(model)
    dense_macs = matmul_macs(model, n_samples, only_active=False)
    sparse_macs = matmul_macs(model, n_samples, only_active=True)
    return {
        "active_params": active,
        "total_params": total,
        "sparsity": 1.0 - active / total,
        "dense_macs": dense_macs,
        "sparse_macs": sparse_macs,
        "mac_reduction": dense_macs / max(sparse_macs, 1),
    }
