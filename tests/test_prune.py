"""Importance criteria, the cubic schedule, and hard-budget pruning with real zeros."""

import numpy as np

from engine import Tensor, softmax_cross_entropy
from nn import Linear, Parameter
from prune import Pruner, cubic_sparsity, magnitude, prunable_weights, saliency
from train import build_mlp


def test_magnitude_is_absolute_value():
    w = Parameter(np.array([[-3.0, 1.0], [0.5, -2.0]]))
    assert np.allclose(magnitude(w), [[3.0, 1.0], [0.5, 2.0]])


def test_saliency_is_abs_weight_times_grad():
    w = Parameter(np.array([[2.0, -1.0]]))
    w.grad = np.array([[0.5, 4.0]])
    assert np.allclose(saliency(w), [[1.0, 4.0]])


def test_cubic_schedule_endpoints_and_shape():
    assert cubic_sparsity(0, 10, 0.9) == 0.0
    assert np.isclose(cubic_sparsity(10, 10, 0.9), 0.9)
    assert cubic_sparsity(5, 10, 0.9) > 0.45  # most of the way there by the midpoint
    vals = [cubic_sparsity(t, 10, 0.9) for t in range(11)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))  # monotonic increasing


def test_pruner_hits_exact_budget_with_genuine_zeros():
    rng = np.random.default_rng(0)
    weights = [Parameter(rng.standard_normal((10, 10))),
               Parameter(rng.standard_normal((5, 8)))]
    pr = Pruner(weights)
    pr.prune_to(0.8, magnitude)
    assert pr.n_pruned() == round(0.8 * 140)
    assert np.isclose(pr.sparsity(), 0.8, atol=1e-9)
    for w in weights:
        assert np.all(w.data[w.mask == 0.0] == 0.0)


def test_pruner_removes_the_smallest_magnitudes():
    w = Parameter(np.array([[1.0, 2.0, 3.0, 4.0]]))
    Pruner([w]).prune_to(0.5, magnitude)
    assert np.allclose(w.mask, [[0.0, 0.0, 1.0, 1.0]])
    assert np.allclose(w.data, [[0.0, 0.0, 3.0, 4.0]])


def test_ranking_is_global_across_layers():
    a = Parameter(np.array([[10.0, 11.0]]))
    b = Parameter(np.array([[0.1, 0.2]]))
    Pruner([a, b]).prune_to(0.5, magnitude)  # the two globally-smallest are both in b
    assert np.allclose(a.mask, [[1.0, 1.0]])
    assert np.allclose(b.mask, [[0.0, 0.0]])


def test_gradual_pruning_is_monotonic():
    w = Parameter(np.random.default_rng(1).standard_normal((20, 20)))
    pr = Pruner([w])
    pr.prune_to(0.5, magnitude)
    dead_at_50 = w.mask == 0.0
    pr.prune_to(0.8, magnitude)
    dead_at_80 = w.mask == 0.0
    assert np.all(dead_at_80[dead_at_50])  # nothing comes back as the target rises
    assert pr.n_pruned() == round(0.8 * 400)


def test_prunable_weights_excludes_biases():
    model = build_mlp(np.random.default_rng(0))
    ws = prunable_weights(model)
    assert len(ws) == 2  # two Linear layers
    assert all(w.data.ndim == 2 for w in ws)


def test_saliency_pruning_on_a_real_model():
    rng = np.random.default_rng(2)
    layer = Linear(8, 4, rng=rng)
    X = Tensor(rng.standard_normal((16, 8)))
    softmax_cross_entropy(layer(X), rng.integers(0, 4, size=16)).backward()
    pr = Pruner(prunable_weights(layer))
    pr.prune_to(0.75, saliency)
    assert pr.n_pruned() == round(0.75 * 32)
    for w in pr.weights:
        assert np.all(w.data[w.mask == 0.0] == 0.0)


def test_saliency_and_magnitude_select_different_masks():
    # Crafted so the criteria disagree: the two largest weights have tiny gradients,
    # so magnitude keeps them while saliency (|w*g|) drops them.
    w_vals = np.array([[1.0, 2.0, 3.0, 4.0]])
    g_vals = np.array([[10.0, 4.0, 0.5, 0.1]])  # |w*g| = [10, 8, 1.5, 0.4]

    w_mag = Parameter(w_vals.copy())
    Pruner([w_mag]).prune_to(0.5, magnitude)

    w_sal = Parameter(w_vals.copy())
    w_sal.grad = g_vals.copy()
    Pruner([w_sal]).prune_to(0.5, saliency)

    # magnitude drops the two smallest |w|, keeping the two largest weights
    assert np.allclose(w_mag.mask, [[0.0, 0.0, 1.0, 1.0]])
    # saliency drops the two smallest |w*g| — the large-but-low-gradient connections
    assert np.allclose(w_sal.mask, [[1.0, 1.0, 0.0, 0.0]])
    # at the same budget the two criteria genuinely disagree
    assert not np.array_equal(w_mag.mask, w_sal.mask)
