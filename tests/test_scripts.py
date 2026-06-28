"""Smoke tests for the runnable entry points: tiny 2-epoch runs into a temp dir.

Artifacts are redirected to pytest's tmp_path by monkeypatching each script's
module-level RESULTS, so the committed results/ files are never touched. We assert
structure and finiteness, not accuracy values (those are pinned by test_sweep.py).
"""

import json

import numpy as np

from train import pareto_run, prune_run, train_mlp


def test_train_mlp_writes_a_finite_learning_curve(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mlp, "RESULTS", tmp_path)
    train_mlp.main(seed=0, epochs=2)

    out = tmp_path / "part2_learning_curve.json"
    assert out.exists()
    data = json.loads(out.read_text())
    for key in ("loss", "train_acc", "test_acc"):
        assert len(data[key]) == 2  # one entry per epoch
        assert all(np.isfinite(v) for v in data[key])
    assert (tmp_path / "part2_learning_curve.png").exists()


def test_prune_run_writes_cost_and_accuracy_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(prune_run, "RESULTS", tmp_path)
    prune_run.main(seed=0, final_sparsity=0.9, epochs=2)

    out = tmp_path / "part3_pruning.json"
    assert out.exists()
    data = json.loads(out.read_text())
    for key in ("pruned_test_acc", "accuracy_cost", "achieved_sparsity",
                "active_params", "total_params", "mac_reduction"):
        assert key in data
    assert np.isfinite(data["pruned_test_acc"])
    assert np.isfinite(data["accuracy_cost"])
    assert data["total_params"] == 9472  # architecture constant, not a trained value


def test_pareto_run_writes_summary_cells_and_claim(tmp_path, monkeypatch):
    # Smallest viable sweep: 2 sparsities x 2 methods x 2 seeds.
    monkeypatch.setattr(pareto_run, "RESULTS", tmp_path)
    monkeypatch.setattr(pareto_run, "SPARSITIES", [0.0, 0.9])
    monkeypatch.setattr(pareto_run, "SEEDS", [0, 1])
    pareto_run.main(epochs=2)

    out = tmp_path / "part4_pareto.json"
    assert out.exists()
    data = json.loads(out.read_text())
    for method in ("saliency", "magnitude"):
        for sparsity in (0.0, 0.9):
            cell = data["summary"][f"{method}@{sparsity}"]
            assert cell["n"] == 2
            assert np.isfinite(cell["mean"]) and np.isfinite(cell["std"])
    assert "claim" in data and np.isfinite(data["claim"]["mean_delta"])
    assert (tmp_path / "part4_pareto.png").exists()
