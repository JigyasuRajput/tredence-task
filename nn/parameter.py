"""A trainable weight that supports masking connections out (hard pruning to zero)."""

import numpy as np

from engine import Tensor


class Parameter(Tensor):
    """A weight tensor plus a 0/1 mask; pruned connections are forced to a true zero."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)
        self.mask = np.ones_like(self.data)  # 1 = live, 0 = pruned

    def masked(self):
        """Effective weight for the forward pass.

        Multiplying by the mask inside the graph means a dead connection both
        contributes zero to the output and receives an exactly zero gradient.
        """
        return self * Tensor(self.mask)

    def apply_mask(self):
        """Force pruned weights back to exactly zero (call after each optimizer step)."""
        self.data *= self.mask
