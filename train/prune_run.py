"""Self-pruning run: train the digits MLP to a target sparsity and report the cost (Part 3).

Run: uv run python -m train.prune_run
"""

import json
from pathlib import Path

import numpy as np

from nn import Adam
from prune import Pruner, prunable_weights, saliency
from train.data import load_digits_data
from train.loop import train
from train.model import build_mlp
from train.prune_loop import train_pruned
from utils import set_seed

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(seed=0, final_sparsity=0.9, epochs=40, lr=0.01):
    set_seed(seed)
    X_train, y_train, X_test, y_test = load_digits_data(seed=seed)

    # Dense baseline at the same seed, so the accuracy cost is apples-to-apples.
    dense = build_mlp(np.random.default_rng(seed))
    dense_hist = train(dense, Adam(dense.parameters(), lr=lr),
                       X_train, y_train, X_test, y_test, epochs=epochs, seed=seed)
    dense_acc = dense_hist["test_acc"][-1]

    # Self-pruned model pruned to the target during training.
    model = build_mlp(np.random.default_rng(seed))
    pruner = Pruner(prunable_weights(model))
    hist = train_pruned(model, Adam(model.parameters(), lr=lr), pruner,
                        X_train, y_train, X_test, y_test,
                        final_sparsity=final_sparsity, importance=saliency,
                        epochs=epochs, seed=seed)
    pruned_acc = hist["test_acc"][-1]
    achieved = pruner.sparsity()

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "part3_pruning.json", "w") as f:
        json.dump({"seed": seed, "target_sparsity": final_sparsity,
                   "achieved_sparsity": achieved, "dense_test_acc": dense_acc,
                   "pruned_test_acc": pruned_acc, "accuracy_cost": dense_acc - pruned_acc,
                   **hist}, f, indent=2)

    print(f"dense test acc {dense_acc:.4f}  ->  "
          f"pruned ({achieved:.1%}) test acc {pruned_acc:.4f}  "
          f"(cost {dense_acc - pruned_acc:+.4f})")


if __name__ == "__main__":
    main()
