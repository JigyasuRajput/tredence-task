"""Reproducibility helpers: pin every RNG we touch from one seed."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Pin Python, NumPy, and the hash seed so a run reproduces exactly."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Return an isolated NumPy Generator, so one stream can't perturb another."""
    return np.random.default_rng(seed)
