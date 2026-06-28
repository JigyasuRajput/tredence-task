"""Weight initialization: correct variances, and signal that survives depth without NaNs."""

import numpy as np

from engine import Tensor
from nn import Linear
from nn.init import he_normal, xavier_normal


def test_he_normal_has_expected_std():
    W = he_normal(256, 256, np.random.default_rng(0))
    assert np.isclose(np.std(W), np.sqrt(2.0 / 256), rtol=0.1)


def test_xavier_normal_has_expected_std():
    W = xavier_normal(100, 300, np.random.default_rng(0))
    assert np.isclose(np.std(W), np.sqrt(2.0 / 400), rtol=0.1)


def test_bias_starts_at_zero():
    assert np.all(Linear(4, 3, rng=np.random.default_rng(0)).bias.data == 0.0)


def test_he_is_larger_than_xavier_on_square_layers():
    he = Linear(256, 256, rng=np.random.default_rng(1), init="he")
    xa = Linear(256, 256, rng=np.random.default_rng(1), init="xavier")
    assert np.std(he.weight.data) > np.std(xa.weight.data)


def _forward_std_through_relu_stack(init, depth=8, width=128, seed=0):
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal((64, width)))
    for _ in range(depth):
        x = Linear(width, width, rng=rng, init=init)(x).relu()
    return float(np.std(x.data))


def test_he_init_keeps_signal_alive_through_depth():
    # The point of He: pre-activation variance stays ~constant, so an 8-layer ReLU
    # stack neither explodes nor vanishes, and stays finite.
    s = _forward_std_through_relu_stack("he")
    assert np.isfinite(s)
    assert 0.2 < s < 3.0


def test_tiny_init_collapses_signal_by_contrast():
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((64, 128)))
    for _ in range(8):
        layer = Linear(128, 128, rng=rng)
        layer.weight.data *= 0.01  # deliberately far too small
        x = layer(x).relu()
    assert np.std(x.data) < 1e-3
