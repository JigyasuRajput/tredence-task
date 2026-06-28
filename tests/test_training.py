"""Dataset loading and a short, stable, NaN-free training run."""

import numpy as np

from engine import Tensor
from nn import Adam
from train import build_mlp, load_digits_data, train
from utils import set_seed


def test_data_shapes_and_standardization():
    X_train, y_train, X_test, y_test = load_digits_data(seed=0)
    assert X_train.shape[1] == 64 and X_test.shape[1] == 64
    assert len(X_train) + len(X_test) == 1797
    assert y_train.min() >= 0 and y_train.max() <= 9
    # Standardized with train stats: zero mean, unit variance on the active pixels.
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-6)
    stds = X_train.std(axis=0)
    assert stds.max() <= 1.0 + 1e-6
    assert (stds > 0.99).sum() >= 40  # all-zero border pixels stay flat; the rest scale to 1


def test_build_mlp_shape_and_param_count():
    model = build_mlp(np.random.default_rng(0))
    out = model(Tensor(np.zeros((5, 64))))
    assert out.shape == (5, 10)
    # two linears: (64x128 + 128) + (128x10 + 10) = 9610 trainable values
    total = sum(p.size for p in model.parameters())
    assert total == 64 * 128 + 128 + 128 * 10 + 10


def test_short_training_run_is_stable_and_accurate():
    set_seed(0)
    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = load_digits_data(seed=0)
    model = build_mlp(rng)
    opt = Adam(model.parameters(), lr=0.01)
    history = train(model, opt, X_train, y_train, X_test, y_test, epochs=10, seed=0)

    assert all(np.isfinite(v) for v in history["loss"])  # no NaNs
    assert history["loss"][-1] < history["loss"][0]       # loss came down
    assert history["test_acc"][-1] > 0.85                 # digits is easy; we should clear this
