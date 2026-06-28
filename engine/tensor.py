"""The Tensor: wraps a NumPy array and remembers how it was made so we can backprop."""

from __future__ import annotations

import numpy as np

# Everything would run in float64. Gradient checking needs the precision, and
# the models here are tiny, so we never pay for it.
DEFAULT_DTYPE = np.float64


class Tensor:
    """A NumPy array plus its gradient and the recipe that produced it."""

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=DEFAULT_DTYPE)
        self.requires_grad = bool(requires_grad)
        # Gradients accumulate into here (Trap E); they start at zero and the
        # backward pass re-zeros the whole graph before each run.
        self.grad = np.zeros_like(self.data)
        # Graph bookkeeping: who fed into me, what op made me, and the closure
        # that pushes my grad back to my parents. Ops fill the closure in; a
        # leaf keeps the no-op.
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    # -- shape passthroughs -
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def zero_grad(self):
        """Reset this tensor's accumulated gradient back to zero."""
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        op = f", op={self._op!r}" if self._op else ""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{op})"
