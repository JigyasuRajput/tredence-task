"""Stability and gradients for softmax and the fused cross-entropy loss."""

import numpy as np

from engine import Tensor, softmax, softmax_cross_entropy
from grad_helpers import check_single_op


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).standard_normal((5, 4)))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p > 0)


def test_softmax_stable_with_huge_logits():
    # Naive exp would overflow here; subtracting the row max keeps it finite.
    p = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_loss_equals_log_c_for_uniform_logits():
    loss = softmax_cross_entropy(Tensor(np.zeros((3, 5))), [0, 1, 2])
    assert np.isclose(loss.data, np.log(5))


def test_loss_finite_when_correct_logit_is_huge():
    loss = softmax_cross_entropy(Tensor([[0.0, 1000.0]], requires_grad=True), [1])
    assert np.isfinite(loss.data)
    assert loss.data < 1e-6


def test_fused_gradient_is_probs_minus_onehot_over_n():
    rng = np.random.default_rng(1)
    targets = np.array([0, 2, 1, 0])
    logits = Tensor(rng.standard_normal((4, 3)), requires_grad=True)
    loss = softmax_cross_entropy(logits, targets)
    loss.grad = np.array(1.0)
    loss._backward()

    expected = softmax(logits.data)
    expected[np.arange(4), targets] -= 1.0
    expected /= 4
    assert np.allclose(logits.grad, expected)


def test_softmax_ce_grad_matches_finite_differences():
    rng = np.random.default_rng(2)
    targets = np.array([0, 2, 1, 0, 3])
    logits = Tensor(rng.standard_normal((5, 4)), requires_grad=True)
    check_single_op(lambda z: softmax_cross_entropy(z, targets), [logits])
