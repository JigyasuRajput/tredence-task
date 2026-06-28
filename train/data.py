"""Load sklearn's digits dataset (data only) and standardize it with NumPy."""

import numpy as np
from sklearn.datasets import load_digits


def load_digits_data(test_fraction=0.2, seed=0):
    """Return standardized (X_train, y_train, X_test, y_test) for the 8x8 digits."""
    digits = load_digits()
    X = digits.data.astype(np.float64)   # (1797, 64) pixel intensities
    y = digits.target.astype(np.int64)   # (1797,) labels 0-9

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_test = int(len(X) * test_fraction)
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    # Standardize using train statistics only, so the test split can't leak in.
    # The epsilon keeps always-zero border pixels finite (they map to a flat 0).
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, y_train, X_test, y_test
