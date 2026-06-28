"""Self-pruning machinery: masks, importance criteria, schedules (Part 3)."""

from .importance import magnitude, saliency
from .pruner import Pruner, prunable_weights
from .schedule import cubic_sparsity

__all__ = ["magnitude", "saliency", "Pruner", "prunable_weights", "cubic_sparsity"]
