"""Multi-seed sparsity sweep comparing saliency vs magnitude pruning."""

import numpy as np

from nn import Adam
from prune import Pruner, magnitude, prunable_weights, saliency
from train.cost import measure_cost
from train.data import load_digits_data
from train.model import build_mlp
from train.prune_loop import train_pruned
from utils import set_seed

CRITERIA = {"magnitude": magnitude, "saliency": saliency}


def run_one(method, target_sparsity, seed, epochs=30, lr=0.01):
    """Train one pruned model and return its final accuracy, sparsity, and cost."""
    set_seed(seed)
    X_train, y_train, X_test, y_test = load_digits_data(seed=seed)
    model = build_mlp(np.random.default_rng(seed))
    pruner = Pruner(prunable_weights(model))
    history = train_pruned(model, Adam(model.parameters(), lr=lr), pruner,
                           X_train, y_train, X_test, y_test,
                           final_sparsity=target_sparsity, importance=CRITERIA[method],
                           epochs=epochs, seed=seed)
    cost = measure_cost(model, n_samples=len(X_test))
    return {"method": method, "target_sparsity": target_sparsity, "seed": seed,
            "test_acc": history["test_acc"][-1],
            "achieved_sparsity": pruner.sparsity(),
            "mac_reduction": cost["mac_reduction"]}


def pareto_sweep(sparsities, methods=("magnitude", "saliency"), seeds=(0, 1, 2, 3, 4),
                 epochs=30):
    """Run every (method, sparsity, seed) combination; return the flat list of results."""
    results = []
    for seed in seeds:
        for method in methods:
            for s in sparsities:
                results.append(run_one(method, s, seed, epochs=epochs))
    return results


def summarize(results):
    """Mean and std of test accuracy across seeds, keyed by (method, target_sparsity)."""
    summary = {}
    for method, s in sorted({(r["method"], r["target_sparsity"]) for r in results}):
        accs = [r["test_acc"] for r in results
                if r["method"] == method and r["target_sparsity"] == s]
        summary[(method, s)] = {"mean": float(np.mean(accs)),
                                "std": float(np.std(accs)), "n": len(accs)}
    return summary


def paired_deltas(results, sparsity):
    """Per-seed (saliency − magnitude) accuracy differences at one sparsity level."""
    by_seed = {}
    for r in results:
        if r["target_sparsity"] == sparsity:
            by_seed.setdefault(r["seed"], {})[r["method"]] = r["test_acc"]
    return np.array([d["saliency"] - d["magnitude"]
                     for d in by_seed.values() if {"saliency", "magnitude"} <= d.keys()])
