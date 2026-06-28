"""Pareto sweep: accuracy vs sparsity for saliency and magnitude pruning (Part 4).

Run: uv run python -m train.pareto_run
Fixed seeds make every number here reproducible from a clean clone.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.sweep import paired_deltas, pareto_sweep, summarize

RESULTS = Path(__file__).resolve().parent.parent / "results"
SPARSITIES = [0.0, 0.5, 0.75, 0.9, 0.95]
SEEDS = [0, 1, 2, 3, 4]


def _claim(results, sparsity):
    """A falsifiable, paired comparison at one sparsity level."""
    deltas = paired_deltas(results, sparsity)
    mean, std = float(np.mean(deltas)), float(np.std(deltas))
    stderr = std / np.sqrt(len(deltas))
    real = bool(abs(mean) > 2 * stderr)  # ~2 standard errors: distinguishable from seed noise
    return {"sparsity": sparsity, "mean_delta": mean, "std_delta": std,
            "stderr": stderr, "n": len(deltas), "distinguishable_from_noise": real}


def main(epochs=30):
    results = pareto_sweep(SPARSITIES, seeds=SEEDS, epochs=epochs)
    summary = summarize(results)
    claim = _claim(results, max(SPARSITIES))

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "part4_pareto.json", "w") as f:
        json.dump({"sparsities": SPARSITIES, "seeds": SEEDS, "epochs": epochs,
                   "runs": results,
                   "summary": {f"{m}@{s}": v for (m, s), v in summary.items()},
                   "claim": claim}, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, color in [("magnitude", "tab:orange"), ("saliency", "tab:blue")]:
        means = [summary[(method, s)]["mean"] for s in SPARSITIES]
        stds = [summary[(method, s)]["std"] for s in SPARSITIES]
        ax.errorbar(SPARSITIES, means, yerr=stds, marker="o", capsize=4,
                    label=method, color=color)
    ax.set(xlabel="target sparsity", ylabel="test accuracy",
           title=f"Sparsity-accuracy Pareto (mean +/- std over {len(SEEDS)} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "part4_pareto.png", dpi=120)

    for s in SPARSITIES:
        sal, mag = summary[("saliency", s)], summary[("magnitude", s)]
        print(f"sparsity {s:>5.0%}:  saliency {sal['mean']:.4f}+/-{sal['std']:.4f}   "
              f"magnitude {mag['mean']:.4f}+/-{mag['std']:.4f}")
    verdict = "distinguishable from noise" if claim["distinguishable_from_noise"] else "within seed noise"
    print(f"\nclaim @ {claim['sparsity']:.0%}: saliency - magnitude = "
          f"{claim['mean_delta']:+.4f} (stderr {claim['stderr']:.4f}, {verdict})")


if __name__ == "__main__":
    main()
