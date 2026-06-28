"""The backward() traversal: topo order, grad reset, accumulation across reused nodes."""

import numpy as np

from engine import Tensor, softmax_cross_entropy
from grad_helpers import numeric_grad


def test_reused_node_accumulates_across_paths():
    # x feeds three ops; its grad must sum all paths: d/dx (x^2 + x) = 2x + 1.
    x = Tensor([2.0, 3.0], requires_grad=True)
    (x * x + x).sum().backward()
    assert np.allclose(x.grad, 2 * x.data + 1)


def test_backward_resets_grad_each_call():
    x = Tensor([2.0, 3.0], requires_grad=True)
    loss = (x * x).sum()
    loss.backward()
    first = x.grad.copy()
    x.grad = np.array([999.0, 999.0])  # stale junk from a hypothetical earlier pass
    loss.backward()
    assert np.allclose(x.grad, first)  # overwritten cleanly, not added on top


def test_diamond_graph_sums_both_branches():
    a = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    branch1 = a.relu()
    branch2 = a * 2.0
    (branch1 + branch2).sum().backward()
    # d/da of relu(a) + 2a = (a > 0) + 2
    assert np.allclose(a.grad, (a.data > 0) + 2.0)


def test_full_graph_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = Tensor(rng.standard_normal((6, 4)))
    W = Tensor(rng.standard_normal((4, 3)), requires_grad=True)
    b = Tensor(rng.standard_normal(3), requires_grad=True)
    targets = rng.integers(0, 3, size=6)

    softmax_cross_entropy(X @ W + b, targets).backward()

    def loss_of(Wd, bd):
        return float(softmax_cross_entropy(X @ Tensor(Wd) + Tensor(bd), targets).data)

    gW = numeric_grad(lambda Wd: loss_of(Wd, b.data), W.data.copy())
    gb = numeric_grad(lambda bd: loss_of(W.data, bd), b.data.copy())
    assert np.allclose(W.grad, gW, atol=1e-6)
    assert np.allclose(b.grad, gb, atol=1e-6)
