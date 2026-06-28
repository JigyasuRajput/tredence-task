"""The masked-weight correctness requirement: pruned connections stay a true zero.

Zeroing a pruned weight once is not enough. With momentum, the optimizer keeps a
running average of past gradients, so a connection that was live before being
pruned carries stale momentum that pushes its weight back off zero — unless we
both re-assert the mask and clear the optimizer state on dead connections.
"""

import numpy as np

from engine import Tensor
from nn import Parameter

DEAD = [(0, 1), (1, 3), (3, 4)]


def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    W = Parameter(rng.standard_normal((4, 5)))
    for i, j in DEAD:
        W.mask[i, j] = 0.0
    x = Tensor(rng.standard_normal((6, 4)))
    Y = Tensor(rng.standard_normal((6, 5)))
    return W, x, Y


def run_momentum(W, x, Y, steps=30, lr=0.05, beta=0.9, discipline=True, stale=True):
    """A toy momentum-SGD loop, optionally with the masking discipline applied."""
    W.apply_mask()  # pruning zeros the dead weights up front
    v = np.zeros_like(W.data)
    if stale:
        # Momentum left over from when the dead connections were still live.
        v[W.mask == 0.0] = 5.0
    for _ in range(steps):
        diff = (x @ W.masked()) - Y
        (diff * diff).sum().backward()
        v = beta * v + W.grad
        W.data -= lr * v
        if discipline:
            W.apply_mask()      # re-assert the hard zero
            v = v * W.mask      # stop momentum accumulating on dead connections
    return v


def test_masked_forward_gives_dead_connections_zero_gradient():
    W, x, Y = make_problem()
    W.apply_mask()
    dead = W.mask == 0.0
    diff = (x @ W.masked()) - Y
    (diff * diff).sum().backward()
    assert np.all(W.grad[dead] == 0.0)
    assert np.any(W.grad[~dead] != 0.0)  # live connections still get real gradients


def test_stale_momentum_drifts_pruned_weight_without_discipline():
    # The bug: even with a clean zero gradient, leftover momentum walks the weight
    # off zero when we forget to clear the optimizer state.
    W, x, Y = make_problem()
    dead = W.mask == 0.0
    run_momentum(W, x, Y, discipline=False)
    assert np.abs(W.data[dead]).max() > 0.1


def test_pruned_weight_is_exactly_zero_at_every_step():
    W, x, Y = make_problem()
    dead = W.mask == 0.0
    W.apply_mask()
    v = np.zeros_like(W.data)
    v[dead] = 5.0
    for _ in range(30):
        diff = (x @ W.masked()) - Y
        (diff * diff).sum().backward()
        v = 0.9 * v + W.grad
        W.data -= 0.05 * v
        W.apply_mask()
        v = v * W.mask
        assert np.all(W.data[dead] == 0.0)  # never even momentarily nonzero


def test_live_weights_still_train_under_discipline():
    W, x, Y = make_problem()
    live = W.mask == 1.0
    W.apply_mask()
    diff0 = (x @ W.masked()) - Y
    loss_before = float((diff0 * diff0).sum().data)
    init_live = W.data[live].copy()
    run_momentum(W, x, Y, discipline=True)
    diff1 = (x @ W.masked()) - Y
    loss_after = float((diff1 * diff1).sum().data)
    assert loss_after < loss_before                       # training actually happened
    assert not np.allclose(W.data[live], init_live)       # live weights moved


def test_dead_state_does_not_leak_into_live_weights():
    # Live weights end up identical whether or not the dead connections carried
    # stale momentum — so dead optimizer state cannot corrupt the live ones.
    Wa, xa, Ya = make_problem(0)
    Wb, xb, Yb = make_problem(0)
    run_momentum(Wa, xa, Ya, discipline=True, stale=True)
    run_momentum(Wb, xb, Yb, discipline=True, stale=False)
    live = Wa.mask == 1.0
    assert np.allclose(Wa.data[live], Wb.data[live])
    assert np.all(Wa.data[~live] == 0.0)
    assert np.all(Wb.data[~live] == 0.0)
