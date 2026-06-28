"""Hand-written reverse-mode autodiff engine (Part 1)."""

from .losses import softmax, softmax_cross_entropy
from .tensor import Tensor

__all__ = ["Tensor", "softmax", "softmax_cross_entropy"]
