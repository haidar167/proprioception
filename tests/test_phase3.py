"""Tests for Phase 3: covariate drift and interoception vs proprioception signals."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from proprioception.data import PermutedDataset, get_permuted_loader
from proprioception.drift import compute_interoceptive_signals
from proprioception.models import BaselineMLP


def test_permuted_dataset():
    x = torch.arange(10, dtype=torch.float32).unsqueeze(0)  # shape (1, 10)
    y = torch.tensor([0])
    base_ds = TensorDataset(x, y)

    perm = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    p_ds = PermutedDataset(base_ds, perm)

    p_x, p_y = p_ds[0]
    assert p_x[0] == 9.0
    assert p_x[9] == 0.0
    assert p_y.item() == 0


def test_compute_interoceptive_signals():
    model = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    x = torch.randn(20, 64)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)

    signals = compute_interoceptive_signals(model, loader, max_batches=2)
    assert "mean_entropy" in signals
    assert "mean_max_confidence" in signals
    assert "mean_h1_norm" in signals
    assert "mean_h2_norm" in signals
    assert "accuracy" in signals
    assert signals["mean_entropy"] >= 0.0
    assert 0.0 <= signals["mean_max_confidence"] <= 1.0
