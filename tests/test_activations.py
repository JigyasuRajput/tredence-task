"""Forward values and finite-difference gradients for ReLU and tanh."""

import numpy as np

from engine import Tensor
from grad_helpers import check_single_op


def randn(shape, seed):
    return Tensor(np.random.default_rng(seed).standard_normal(shape), requires_grad=True)


def test_relu_forward():
    x = Tensor([-2.0, -0.1, 0.0, 0.5, 3.0])
    assert np.allclose(x.relu().data, [0.0, 0.0, 0.0, 0.5, 3.0])


def test_relu_grad_is_gated_by_sign():
    x = Tensor([-2.0, 0.5, 3.0], requires_grad=True)
    out = x.relu()
    out.grad = np.array([1.0, 1.0, 1.0])
    out._backward()
    # Grad passes through where the input was positive, blocked where negative.
    assert np.allclose(x.grad, [0.0, 1.0, 1.0])


def test_relu_grad_matches_finite_differences():
    check_single_op(lambda a: a.relu(), [randn((4, 5), 0)])


def test_tanh_forward():
    x = Tensor([-1.0, 0.0, 1.0])
    assert np.allclose(x.tanh().data, np.tanh([-1.0, 0.0, 1.0]))


def test_tanh_grad_matches_finite_differences():
    check_single_op(lambda a: a.tanh(), [randn((4, 5), 1)])
