"""Weight initializers: He for ReLU paths, Xavier (Glorot) for linear / tanh paths."""

import numpy as np


def he_normal(fan_in, fan_out, rng):
    """Std = sqrt(2/fan_in). ReLU zeros ~half its inputs, halving variance; the 2 restores it."""
    std = np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_in, fan_out)) * std


def xavier_normal(fan_in, fan_out, rng):
    """Std = sqrt(2/(fan_in+fan_out)). Balances forward-activation and backward-gradient variance."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.standard_normal((fan_in, fan_out)) * std


_INITS = {"he": he_normal, "xavier": xavier_normal}


def init_weight(fan_in, fan_out, rng, kind="he"):
    """Build a weight matrix using the named initializer."""
    return _INITS[kind](fan_in, fan_out, rng)
