"""Regrowth: the revival signal, topology swaps, and clean optimizer revival."""

import numpy as np

from engine import Tensor, softmax_cross_entropy
from nn import Adam, Linear, Parameter
from prune import Pruner, prunable_weights, prune_and_grow, saliency


def test_dead_connection_has_zero_update_grad_but_nonzero_revival_signal():
    # A pruned weight's *update* gradient is zero, but the gradient it would have
    # had if active (the revival signal) is not — the two must stay separate.
    rng = np.random.default_rng(0)
    layer = Linear(6, 4, rng=rng)
    layer.weight.mask[0, 0] = 0.0
    layer.weight.apply_mask()
    X = Tensor(rng.standard_normal((12, 6)))
    softmax_cross_entropy(layer(X), rng.integers(0, 4, size=12)).backward()

    assert layer.weight.grad[0, 0] == 0.0           # update gradient: frozen
    assert layer.weight.dense_grad()[0, 0] != 0.0   # revival signal: alive


def test_prune_and_grow_keeps_sparsity_and_swaps_topology():
    w = Parameter(np.array([[5.0, 0.0, 4.0, 0.0]]))
    w.mask = np.array([[1.0, 0.0, 1.0, 0.0]])  # cols 1 and 3 are dead
    grow_signal = np.array([[0.0, 9.0, 0.0, 0.1]])  # dead col 1 looks far more useful

    swapped = prune_and_grow(
        [w], fraction=0.5,
        drop_score=lambda x: np.abs(x.data),   # weakest live is col 2 (4 < 5)
        grow_score=lambda x: grow_signal,      # strongest dead is col 1
    )

    assert swapped == 1
    assert np.allclose(w.mask, [[1.0, 1.0, 0.0, 0.0]])  # kept 0, grew 1, dropped 2, left 3
    assert (w.mask == 1.0).sum() == 2                   # sparsity unchanged
    assert w.data[0, 1] == 0.0                          # revived weight starts at zero
    assert w.data[0, 2] == 0.0                          # dropped weight is zeroed


def test_revived_connection_starts_clean_under_adam():
    rng = np.random.default_rng(1)
    layer = Linear(5, 3, rng=rng)
    opt = Adam(layer.parameters(), lr=0.01)
    X = Tensor(rng.standard_normal((20, 5)))
    y = rng.integers(0, 3, size=20)

    for _ in range(10):  # warm up the moments
        softmax_cross_entropy(layer(X), y).backward()
        opt.step()

    layer.weight.mask[0, 0] = 0.0  # prune a connection that now has warm moments
    layer.weight.apply_mask()
    softmax_cross_entropy(layer(X), y).backward()
    opt.step()  # the step clears the dead connection's moments

    idx = opt.params.index(layer.weight)
    assert opt.m[idx][0, 0] == 0.0 and opt.v[idx][0, 0] == 0.0

    layer.weight.mask[0, 0] = 1.0  # revive it
    before = layer.weight.data[0, 0]
    softmax_cross_entropy(layer(X), y).backward()
    opt.step()
    # Clean start (zero moments) means a bounded first step, not a stale-v explosion.
    assert abs(layer.weight.data[0, 0] - before) < 0.05


def test_regrowth_during_training_holds_sparsity_and_stays_finite():
    rng = np.random.default_rng(2)
    layer = Linear(8, 4, rng=rng)
    opt = Adam(layer.parameters(), lr=0.01)
    pruner = Pruner(prunable_weights(layer))
    X = Tensor(rng.standard_normal((24, 8)))
    y = rng.integers(0, 4, size=24)

    for _ in range(5):
        softmax_cross_entropy(layer(X), y).backward()
        opt.step()
    pruner.prune_to(0.5, saliency)
    target = pruner.sparsity()

    # One real regrowth step using the would-be-dense gradient as the grow signal.
    softmax_cross_entropy(layer(X), y).backward()
    prune_and_grow(pruner.weights, fraction=0.2,
                   drop_score=lambda w: np.abs(w.data),
                   grow_score=lambda w: np.abs(w.dense_grad()),
                   optimizer=opt)
    assert abs(pruner.sparsity() - target) < 1e-9  # swap kept the budget exactly

    for _ in range(5):  # keep training; nothing should blow up
        softmax_cross_entropy(layer(X), y).backward()
        opt.step()
    assert np.all(np.isfinite(layer.weight.data))
