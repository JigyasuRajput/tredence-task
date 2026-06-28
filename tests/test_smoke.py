"""Smoke test: the skeleton imports and the seed helper is deterministic."""

import importlib

import numpy as np

from utils import make_rng, set_seed


def test_packages_import():
    for name in ("engine", "nn", "prune", "train", "utils"):
        assert importlib.import_module(name) is not None


def test_set_seed_is_reproducible():
    set_seed(0)
    a = np.random.randn(5)
    set_seed(0)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_make_rng_is_isolated_and_reproducible():
    first = make_rng(123).standard_normal(5)
    second = make_rng(123).standard_normal(5)
    assert np.array_equal(first, second)
