"""Dataset loading, the training loop, and the evidence sweeps (Parts 2 & 4)."""

from .cost import active_params, measure_cost, sparse_forward, total_params
from .data import load_digits_data
from .loop import accuracy, train
from .model import build_mlp
from .prune_loop import train_pruned

__all__ = [
    "load_digits_data", "build_mlp", "train", "accuracy", "train_pruned",
    "measure_cost", "sparse_forward", "active_params", "total_params",
]
