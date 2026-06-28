"""The sweep harness: hits targets, reproduces exactly, and summarizes correctly."""

from train.sweep import pareto_sweep, run_one, summarize


def test_run_one_hits_target_and_returns_metrics():
    r = run_one("saliency", 0.8, seed=0, epochs=8)
    assert abs(r["achieved_sparsity"] - 0.8) < 1e-2
    assert 0.0 <= r["test_acc"] <= 1.0
    assert r["mac_reduction"] > 1.0


def test_sweep_is_reproducible_with_fixed_seeds():
    a = run_one("magnitude", 0.9, seed=1, epochs=6)
    b = run_one("magnitude", 0.9, seed=1, epochs=6)
    assert a["test_acc"] == b["test_acc"]
    assert a["achieved_sparsity"] == b["achieved_sparsity"]


def test_summarize_groups_by_method_and_sparsity():
    results = pareto_sweep([0.0, 0.9], methods=("magnitude", "saliency"),
                           seeds=(0, 1), epochs=4)
    summary = summarize(results)
    assert ("saliency", 0.9) in summary and ("magnitude", 0.0) in summary
    assert summary[("magnitude", 0.0)]["n"] == 2
    assert all(0.0 <= v["mean"] <= 1.0 for v in summary.values())
