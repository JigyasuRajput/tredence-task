"""Connection importance scores. Magnitude is the baseline; saliency uses the gradient."""

import numpy as np


def magnitude(weight):
    """Pure magnitude |w|: the trivial baseline criterion."""
    return np.abs(weight.data)


def saliency(weight):
    """First-order saliency |w * dL/dw|: approximates the loss increase from removing w.

    Removing connection i means w_i -> 0, i.e. a step dw_i = -w_i. A first-order
    Taylor expansion gives dL ~= g_i * (-w_i), so the loss change has magnitude
    |w_i * g_i|. Connections with the smallest saliency are the cheapest to drop.
    """
    return np.abs(weight.data * weight.grad)
