"""End-to-end gradual pruning: hit the budget, keep accuracy, keep zeros frozen."""

import numpy as np

from nn import Adam
from prune import Pruner, prunable_weights, saliency
from train import build_mlp, load_digits_data, train_pruned
from utils import set_seed


def test_pruned_training_hits_budget_and_stays_accurate():
    set_seed(0)
    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = load_digits_data(seed=0)
    model = build_mlp(rng)
    pruner = Pruner(prunable_weights(model))
    history = train_pruned(model, Adam(model.parameters(), lr=0.01), pruner,
                           X_train, y_train, X_test, y_test,
                           final_sparsity=0.9, importance=saliency, epochs=25, seed=0)

    # Hard budget: exactly the target count of connections removed.
    assert pruner.n_pruned() == round(0.9 * pruner.total())
    assert abs(pruner.sparsity() - 0.9) < 1e-3

    # Genuine zeros, and they stayed zero through the post-pruning recovery epochs.
    for w in pruner.weights:
        assert np.all(w.data[w.mask == 0.0] == 0.0)

    assert all(np.isfinite(v) for v in history["loss"])
    assert history["test_acc"][-1] > 0.9  # 90% sparse and still well above chance


def test_sparsity_increases_monotonically_to_target():
    set_seed(1)
    rng = np.random.default_rng(1)
    X_train, y_train, X_test, y_test = load_digits_data(seed=1)
    model = build_mlp(rng)
    pruner = Pruner(prunable_weights(model))
    history = train_pruned(model, Adam(model.parameters(), lr=0.01), pruner,
                           X_train, y_train, X_test, y_test,
                           final_sparsity=0.8, importance=saliency, epochs=20, seed=1)

    sp = history["sparsity"]
    assert all(b >= a - 1e-12 for a, b in zip(sp, sp[1:]))  # never un-prunes
    assert abs(sp[-1] - 0.8) < 1e-3
