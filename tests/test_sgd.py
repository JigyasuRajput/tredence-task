"""SGD with momentum: it converges, and it honours the prune mask."""

import numpy as np

from engine import Tensor, softmax_cross_entropy
from nn import Linear, Parameter, Sequential, SGD


def test_sgd_minimizes_a_quadratic():
    target = np.array([1.0, -2.0, 3.0])
    w = Parameter(np.zeros(3))
    opt = SGD([w], lr=0.1, momentum=0.9)
    for _ in range(100):
        diff = w - Tensor(target)
        (diff * diff).sum().backward()
        opt.step()
    assert np.allclose(w.data, target, atol=1e-2)


def test_sgd_trains_a_learnable_classifier_down():
    rng = np.random.default_rng(0)
    X = Tensor(rng.standard_normal((40, 5)))
    y = (X.data @ rng.standard_normal((5, 3))).argmax(axis=1)  # a separable target
    net = Sequential(Linear(5, 3, rng=rng))
    opt = SGD(net.parameters(), lr=0.1, momentum=0.9)

    first = last = None
    for _ in range(200):
        loss = softmax_cross_entropy(net(X), y)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.data)
        last = float(loss.data)
    assert last < first * 0.3


def test_pruned_column_stays_zero_when_pruned_from_the_start():
    rng = np.random.default_rng(1)
    layer = Linear(4, 3, rng=rng)
    layer.weight.mask[:, 0] = 0.0
    layer.weight.apply_mask()
    opt = SGD(layer.parameters(), lr=0.1, momentum=0.9)
    X = Tensor(rng.standard_normal((10, 4)))
    y = rng.integers(0, 3, size=10)
    for _ in range(30):
        softmax_cross_entropy(layer(X), y).backward()
        opt.step()
        assert np.all(layer.weight.data[:, 0] == 0.0)
    velocity = opt.velocity[opt.params.index(layer.weight)]
    assert np.all(velocity[:, 0] == 0.0)


def test_pruning_midtraining_holds_under_momentum():
    # Prune a connection that already carries momentum, then keep training: the
    # optimizer's mask discipline must pin it at exactly zero from then on.
    rng = np.random.default_rng(2)
    layer = Linear(4, 3, rng=rng)
    opt = SGD(layer.parameters(), lr=0.1, momentum=0.9)
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
