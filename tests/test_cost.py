"""Honest cost: the sparse forward matches dense, and FLOPs really drop with sparsity."""

import numpy as np

from engine import Tensor
from prune import Pruner, magnitude, prunable_weights
from train import active_params, build_mlp, measure_cost, sparse_forward, total_params


def test_sparse_forward_matches_dense_forward():
    rng = np.random.default_rng(0)
    model = build_mlp(rng)
    Pruner(prunable_weights(model)).prune_to(0.8, magnitude)
    X = rng.standard_normal((10, 64))
    dense = model(Tensor(X)).data
    sparse = sparse_forward(model, X)
    assert np.allclose(dense, sparse, atol=1e-9)


def test_active_param_count_after_pruning():
    rng = np.random.default_rng(1)
    model = build_mlp(rng)
    total = total_params(model)
    Pruner(prunable_weights(model)).prune_to(0.75, magnitude)
    assert active_params(model) == total - round(0.75 * total)


def test_mac_reduction_tracks_sparsity():
    rng = np.random.default_rng(2)
    model = build_mlp(rng)
    Pruner(prunable_weights(model)).prune_to(0.9, magnitude)
    cost = measure_cost(model, n_samples=100)
    assert cost["sparse_macs"] < cost["dense_macs"]              # genuinely fewer ops
    assert cost["sparsity"] > 0.89
    # dense/sparse MACs equal total/active = 1/(1 - sparsity)
    assert abs(cost["mac_reduction"] - 1.0 / (1.0 - cost["sparsity"])) < 1e-6


def test_sparse_forward_ignores_dead_weight_values():
    # Genuine zeros: scribbling junk into the dead slots of the data array must not
    # change the sparse forward, because it only reads live connections.
    rng = np.random.default_rng(3)
    model = build_mlp(rng)
    Pruner(prunable_weights(model)).prune_to(0.5, magnitude)
    X = rng.standard_normal((8, 64))
    before = sparse_forward(model, X)
    for w in prunable_weights(model):
        w.data[w.mask == 0.0] = 123.456  # only touches dead slots; mask still zero there
    after = sparse_forward(model, X)
    assert np.allclose(before, after)
