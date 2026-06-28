"""Adam: bias correction, convergence, and the same prune-mask discipline as SGD."""

import numpy as np

from engine import Tensor, softmax_cross_entropy
from nn import Adam, Linear, Parameter


def test_first_step_is_about_lr_thanks_to_bias_correction():
    # With bias correction, the first update has magnitude ~lr regardless of the
    # gradient's scale; without it the first step would be tiny.
    w = Parameter(np.array([0.0]))
    opt = Adam([w], lr=0.01)
    w.grad = np.array([7.0])
    opt.step()
    assert np.isclose(abs(w.data[0]), 0.01, rtol=1e-3)


def test_adam_minimizes_a_quadratic():
    target = np.array([1.0, -2.0, 3.0])
    w = Parameter(np.zeros(3))
    opt = Adam([w], lr=0.05)
    for _ in range(500):
        diff = w - Tensor(target)
        (diff * diff).sum().backward()
        opt.step()
    assert np.allclose(w.data, target, atol=1e-2)


def test_adam_trains_a_learnable_classifier_down():
    rng = np.random.default_rng(0)
    X = Tensor(rng.standard_normal((40, 5)))
    y = (X.data @ rng.standard_normal((5, 3))).argmax(axis=1)
    layer = Linear(5, 3, rng=rng, init="xavier")
    opt = Adam(layer.parameters(), lr=0.05)

    first = last = None
    for _ in range(200):
        loss = softmax_cross_entropy(layer(X), y)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.data)
        last = float(loss.data)
    assert last < first * 0.3


def test_adam_pruning_midtraining_holds_and_clears_moments():
    rng = np.random.default_rng(2)
    layer = Linear(4, 3, rng=rng)
    opt = Adam(layer.parameters(), lr=0.05)
    X = Tensor(rng.standard_normal((10, 4)))
    y = rng.integers(0, 3, size=10)

    for _ in range(20):
        softmax_cross_entropy(layer(X), y).backward()
        opt.step()

    layer.weight.mask[0, 0] = 0.0
    layer.weight.apply_mask()
    for _ in range(20):
        softmax_cross_entropy(layer(X), y).backward()
        opt.step()
        assert layer.weight.data[0, 0] == 0.0

    idx = opt.params.index(layer.weight)
    assert opt.m[idx][0, 0] == 0.0
    assert opt.v[idx][0, 0] == 0.0
