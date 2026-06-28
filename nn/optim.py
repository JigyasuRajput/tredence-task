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

    def reset_dead_state(self):
        """Zero the velocity on currently-dead connections (e.g. right after regrowth)."""
        for p, v in zip(self.params, self.velocity):
            v *= p.mask


class Adam(Optimizer):
    """Adam with bias-corrected moments; pruned connections carry no moment state."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]  # first moment
        self.v = [np.zeros_like(p.data) for p in self.params]  # second moment
        self.t = 0  # global step, used for bias correction

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        # Bias correction: early moments start at zero and are biased small, so we
        # divide by (1 - beta**t) to undo it — this makes the first step ~lr.
        bias1 = 1.0 - b1 ** self.t
        bias2 = 1.0 - b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            p.data -= self.lr * (m / bias1) / (np.sqrt(v / bias2) + self.eps)
            # Dead connections: hard zero and no lingering moments, so a revived
            # connection starts cleanly rather than from a corrupt state.
            p.apply_mask()
            m *= p.mask
            v *= p.mask

    def reset_dead_state(self):
        """Zero both moments on currently-dead connections (e.g. right after regrowth)."""
        for p, m, v in zip(self.params, self.m, self.v):
            m *= p.mask
            v *= p.mask
