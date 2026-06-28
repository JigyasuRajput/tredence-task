"""Numerically stable softmax and the fused softmax + cross-entropy loss."""

import numpy as np

from .tensor import Tensor


def softmax(logits):
    """Row-wise softmax of a NumPy array, max subtracted for stability."""
    shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / exp.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, targets):
    """Mean cross-entropy over a batch from logits and integer labels, fused for stability."""
    z = logits.data
    targets = np.asarray(targets)
    n = z.shape[0]
    rows = np.arange(n)

    # Stable log-softmax: subtract the row max so exp never overflows.
    shift = z - z.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shift).sum(axis=1, keepdims=True))
    log_probs = shift - log_sum_exp
    probs = np.exp(log_probs)
    loss_val = -log_probs[rows, targets].mean()

    out = Tensor(loss_val, requires_grad=logits.requires_grad,
                 _children=(logits,), _op="softmax_ce")

    def _backward():
        if logits.requires_grad:
            # Fused gradient in closed form: (softmax - onehot) / N. Both more
            # stable and less error-prone than backprop through softmax then log.
            grad = probs.copy()
            grad[rows, targets] -= 1.0
            grad /= n
            logits.grad += grad * out.grad

    out._backward = _backward
    return out
