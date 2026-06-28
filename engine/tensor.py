"""The Tensor: wraps a NumPy array and remembers how it was made so we can backprop."""

from __future__ import annotations

import numpy as np

# Everything runs in float64. Gradient checking needs the precision, and the
# models here are tiny, so we never pay for it.
DEFAULT_DTYPE = np.float64


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing any broadcasting in the forward op."""
    # NumPy broadcasting can (1) prepend new axes and (2) stretch size-1 axes.
    # Reverse both by summing: first the extra leading axes, then the stretched ones.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _as_tensor(x):
    """Wrap a scalar or array as a constant Tensor; pass an existing Tensor through."""
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """A NumPy array plus its gradient and the recipe that produced it."""

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=DEFAULT_DTYPE)
        self.requires_grad = bool(requires_grad)
        # Gradients accumulate into here; they start at zero and the backward
        # pass re-zeros the whole graph before each run.
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

    # -- elementwise ops --
    # Each op forwards with NumPy and installs a closure that routes the upstream
    # grad to its parents, summing back over any axes NumPy broadcast.
    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="-")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data / other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="/")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(
                    -out.grad * self.data / (other.data ** 2), other.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    # Reflected forms so `scalar + tensor`, `scalar - tensor`, etc. also work.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return _as_tensor(other) - self

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    # -- matmul and reductions --
    def __matmul__(self, other):
        # Targets the 2-D (batch, features) @ (features, units) case the MLP uses.
        other = _as_tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="@")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                # Put back the axes we collapsed, then broadcast the grad over them.
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op="mean")
        # Each input contributed 1/n to the average, so the grad is scaled by 1/n.
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([self.data.shape[ax] for ax in axes]))

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g / n

        out._backward = _backward
        return out

    def __repr__(self):
        op = f", op={self._op!r}" if self._op else ""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{op})"
