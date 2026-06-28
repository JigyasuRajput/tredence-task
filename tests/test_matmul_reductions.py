"""Forward values and finite-difference gradients for matmul, sum, and mean."""

import numpy as np

from engine import Tensor
from grad_helpers import check_single_op


def randn(shape, seed):
    return Tensor(np.random.default_rng(seed).standard_normal(shape), requires_grad=True)


def test_matmul_forward():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose((Tensor(a) @ Tensor(b)).data, a @ b)


def test_matmul_grad():
    a, b = randn((4, 3), 0), randn((3, 5), 1)
    check_single_op(lambda x, y: x @ y, [a, b])


def test_matmul_shared_input_accumulates():
    # x @ x routes grad through both operands; the closure must add, not overwrite.
    x = randn((3, 3), 2)
    check_single_op(lambda a: a @ a, [x])


def test_sum_value():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.isclose(a.sum().data, 21.0)
    assert np.allclose(a.sum(axis=0).data, [5.0, 7.0, 9.0])
    assert np.allclose(a.sum(axis=1, keepdims=True).data, [[6.0], [15.0]])


def test_sum_grads():
    check_single_op(lambda a: a.sum(), [randn((4, 3), 3)])
    check_single_op(lambda a: a.sum(axis=0), [randn((4, 3), 4)])
    check_single_op(lambda a: a.sum(axis=1), [randn((4, 3), 5)])
    check_single_op(lambda a: a.sum(axis=1, keepdims=True), [randn((4, 3), 6)])


def test_mean_value():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.isclose(a.mean().data, 3.5)
    assert np.allclose(a.mean(axis=0).data, [2.5, 3.5, 4.5])


def test_mean_grads():
    check_single_op(lambda a: a.mean(), [randn((4, 3), 7)])
    check_single_op(lambda a: a.mean(axis=0), [randn((4, 3), 8)])
    check_single_op(lambda a: a.mean(axis=1, keepdims=True), [randn((4, 3), 9)])
