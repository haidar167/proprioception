"""Tests for Phase 2: rehearsal buffering and self-repair reflex."""

import torch
from torch.utils.data import TensorDataset

from proprioception.data import create_rehearsal_buffer
from proprioception.models import BaselineMLP
from proprioception.repair import run_repair_step_optimization


def test_rehearsal_buffer_creation():
    dummy_x = torch.randn(100, 64)
    dummy_y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(dummy_x, dummy_y)

    buf_x, buf_y = create_rehearsal_buffer(dataset, buffer_size=25, seed=42)
    assert buf_x.shape == (25, 64)
    assert buf_y.shape == (25,)


def test_targeted_repair_param_freezing():
    model = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    dummy_x = torch.randn(50, 64)
    dummy_y = torch.randint(0, 10, (50,))

    # Target Layer 1 (fc2: 32*16 + 16 = 528 params)
    repaired, elapsed, touched = run_repair_step_optimization(
        model, dummy_x, dummy_y, target_layer=1, num_steps=5, batch_size=8, seed=42
    )

    assert touched == (32 * 16 + 16)
    assert elapsed > 0.0
    # Check that layer 0 weights did not change
    assert torch.equal(model.fc1.weight, repaired.fc1.weight)
    # Check that layer 2 weights did not change
    assert torch.equal(model.fc3.weight, repaired.fc3.weight)
    # Check that layer 1 weights changed
    assert not torch.equal(model.fc2.weight, repaired.fc2.weight)


def test_full_repair_touches_all_params():
    model = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    dummy_x = torch.randn(50, 64)
    dummy_y = torch.randint(0, 10, (50,))

    total_model_params = sum(p.numel() for p in model.parameters())
    repaired, elapsed, touched = run_repair_step_optimization(
        model, dummy_x, dummy_y, target_layer=None, num_steps=5, batch_size=8, seed=42
    )
    assert touched == total_model_params
    assert not torch.equal(model.fc1.weight, repaired.fc1.weight)
    assert not torch.equal(model.fc2.weight, repaired.fc2.weight)
    assert not torch.equal(model.fc3.weight, repaired.fc3.weight)
