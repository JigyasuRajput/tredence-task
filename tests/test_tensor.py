"""The tensor type and its backprop plumbing."""

import numpy as np

from engine import Tensor


def test_wraps_array_as_float64():
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t.dtype == np.float64
    assert np.array_equal(t.data, [[1, 2], [3, 4]])


def test_grad_starts_at_zero_with_matching_shape():
    t = Tensor(np.ones((3, 5)))
    assert t.grad.shape == t.shape
    assert np.all(t.grad == 0.0)


def test_requires_grad_defaults_off_and_is_settable():
    assert Tensor(1.0).requires_grad is False
    assert Tensor(1.0, requires_grad=True).requires_grad is True


def test_remembers_parents_and_op():
    a, b = Tensor(1.0), Tensor(2.0)
    out = Tensor(3.0, _children=(a, b), _op="+")
    assert out._prev == (a, b)
    assert out._op == "+"


def test_backward_closure_accumulates_into_parents():
    # Hand-wire a parent/child link the way an op will, then confirm the closure
    # accumulates (+=) rather than overwrites — needed when a node is reused.
    parent = Tensor(5.0, requires_grad=True)
    child = Tensor(5.0, _children=(parent,), _op="identity")

    def _backward():
        parent.grad += 2.0 * child.grad

    child._backward = _backward

    child.grad = np.array(1.0)
    child._backward()
    child._backward()  # called twice; grad should add up, not reset
    assert parent.grad == 4.0


def test_zero_grad_resets():
    t = Tensor(np.ones(4))
    t.grad = np.ones(4)
    t.zero_grad()
    assert np.all(t.grad == 0.0)


def test_repr_mentions_shape():
    assert "shape=(2,)" in repr(Tensor([1.0, 2.0]))
