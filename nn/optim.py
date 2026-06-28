"""From-scratch optimizers that keep pruned connections frozen at zero."""

import numpy as np


class Optimizer:
    """Holds the parameters and a shared way to clear their gradients."""

    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    """SGD with classical (heavy-ball) momentum: v = m*v + g, then w -= lr*v."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v
            # Keep pruned connections a hard zero and stop momentum from living
            # on them — otherwise stale velocity would walk them back off zero.
            p.apply_mask()
            v *= p.mask
