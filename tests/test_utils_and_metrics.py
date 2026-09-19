"""Unit tests for utils, metrics, and models."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from proprioception.metrics import (
    compute_auc,
    compute_classification_metrics,
    compute_recovery_percentage,
)
from proprioception.models import BaselineMLP, evaluate_model
from proprioception.utils import get_git_commit_hash, save_results, set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    t1 = torch.randn(5, 5)
    set_seed(42)
    t2 = torch.randn(5, 5)
    assert torch.equal(t1, t2)


def test_metrics_computation():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    auc = compute_auc(y_true, y_scores)
    assert auc == 1.0

    y_pred = [0, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_recovery_percentage():
    # 90% healthy, dropped to 50% on damage, repaired to 80%
    # recovery = (80 - 50) / (90 - 50) = 30 / 40 = 75%
    rec = compute_recovery_percentage(0.90, 0.50, 0.80)
    assert pytest.approx(rec, 0.01) == 75.0

    # No recovery: repaired is still 50%
    rec_none = compute_recovery_percentage(0.90, 0.50, 0.50)
    assert pytest.approx(rec_none, 0.01) == 0.0

    # Full recovery: repaired is 90%
    rec_full = compute_recovery_percentage(0.90, 0.50, 0.90)
    assert pytest.approx(rec_full, 0.01) == 100.0


def test_save_results_structure():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_file = Path(tmp_dir) / "test_res.json"
        saved = save_results({"metric_a": 0.95}, res_file, seed=42)
        assert "timestamp" in saved
        assert "seed" in saved
        assert "git_commit" in saved
        assert saved["metric_a"] == 0.95
        assert res_file.exists()


def test_baseline_mlp_forward():
    model = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    x = torch.randn(8, 64)
    out = model(x)
    assert out.shape == (8, 10)
    logits, acts = model.forward_with_activations(x)
    assert logits.shape == (8, 10)
    assert len(acts) == 2
    assert acts[0].shape == (8, 32)
    assert acts[1].shape == (8, 16)
