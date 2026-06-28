"""A trainable weight that supports masking connections out (hard pruning to zero)."""

import numpy as np

from engine import Tensor


class Parameter(Tensor):
    """A weight tensor plus a 0/1 mask; pruned connections are forced to a true zero."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)
        self.mask = np.ones_like(self.data)  # 1 = live, 0 = pruned
        self._effective = None               # last masked() product, for the revival signal

    def masked(self):
        """Effective weight for the forward pass.

        Multiplying by the mask inside the graph means a dead connection both
        contributes zero to the output and receives an exactly zero gradient.
        """
        self._effective = self * Tensor(self.mask)
        return self._effective

    def apply_mask(self):
        """Force pruned weights back to exactly zero (call after each optimizer step)."""
        self.data *= self.mask

    def dense_grad(self):
        """The gradient ignoring the mask — the revival signal for dead connections.

        After backward(), the masked() product's grad holds dL/d(W*mask), i.e. dL/dW
        evaluated on the current sparse activations. For a live connection this is the
        ordinary gradient; for a dead one it is the gradient it WOULD have had if active,
        which is what we score regrowth on. This is kept separate from `grad` (the update
        gradient, which is zero on dead connections), so the revival signal never moves a
        pruned weight.
        """
        return self._effective.grad
