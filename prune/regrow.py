"""Regrowth: let dead connections come back when they look useful, keeping sparsity fixed."""

import numpy as np


def prune_and_grow(weights, fraction, drop_score, grow_score, optimizer=None):
    """RigL-style topology update: drop the weakest live connections and grow the
    strongest dead ones, keeping the number of active connections constant.

    `drop_score(w)` scores live connections (smallest are dropped); `grow_score(w)`
    scores dead connections (largest are grown, e.g. |would-be-dense gradient|).
    `fraction` is the share of currently-live connections to swap. If an optimizer is
    given, its state on dead connections is reset so revived weights start clean.
    Returns the number of connections swapped.
    """
    flat_mask = np.concatenate([w.mask.ravel() for w in weights])
    drop = np.concatenate([drop_score(w).ravel() for w in weights])
    grow = np.concatenate([grow_score(w).ravel() for w in weights])

    live_idx = np.flatnonzero(flat_mask == 1.0)
    dead_idx = np.flatnonzero(flat_mask == 0.0)
    k = min(int(round(fraction * len(live_idx))), len(live_idx), len(dead_idx))
    if k == 0:
        return 0

    drop_idx = live_idx[np.argsort(drop[live_idx])[:k]]            # weakest live
    grow_idx = dead_idx[np.argsort(grow[dead_idx])[::-1][:k]]      # strongest dead

    new_mask = flat_mask.copy()
    new_mask[drop_idx] = 0.0
    new_mask[grow_idx] = 1.0

    offset = 0
    for w in weights:
        n = w.mask.size
        w.mask = new_mask[offset:offset + n].reshape(w.mask.shape).copy()
        # Dropped weights -> 0; grown weights were already 0, so they start fresh.
        w.apply_mask()
        offset += n

    if optimizer is not None:
        optimizer.reset_dead_state()
    return k
