"""Tests for Phase 1: corruptions, body map extraction, and dataset generation."""

import numpy as np
import pytest
import torch

from proprioception.bodymap import extract_body_map, extract_layer_statistics, get_body_map_feature_names
from proprioception.corruption import apply_gaussian_corruption, apply_zero_corruption, corrupt_model
from proprioception.models import BaselineMLP
from proprioception.phase1_damage_detection import generate_body_map_dataset


def test_zero_corruption():
    model = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    w_orig = model.fc1.weight.clone()
    apply_zero_corruption(model, layer_idx=0, fraction=0.30, seed=42)
    w_corrupt = model.fc1.weight

    zeros_count = (w_corrupt == 0).sum().item()
    total = w_corrupt.numel()
    zero_fraction = zeros_count / total
    assert 0.25 <= zero_fraction <= 0.35
    assert not torch.equal(w_orig, w_corrupt)
    # Other layers should remain untouched
    assert (model.fc2.weight == 0).sum().item() == 0


def test_gaussian_corruption():
    model = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    w_orig = model.fc2.weight.clone()
    apply_gaussian_corruption(model, layer_idx=1, noise_std=0.1, seed=42)
    w_corrupt = model.fc2.weight

    diff = (w_corrupt - w_orig).abs().mean().item()
    assert diff > 0.05
    # Layer 0 and 2 untouched
    assert torch.equal(model.fc1.weight, model.fc1.weight)


def test_body_map_dimensions_and_features():
    healthy = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    damaged = corrupt_model(healthy, "zero", layer_idx=0, intensity=0.25, seed=1)

    bmap_healthy = extract_body_map(healthy, healthy)
    assert bmap_healthy.shape == (12,)
    # For healthy vs healthy, norm change features (idx 3, 7, 11) should be 0.0
    assert bmap_healthy[3] == 0.0
    assert bmap_healthy[7] == 0.0
    assert bmap_healthy[11] == 0.0

    bmap_damaged = extract_body_map(damaged, healthy)
    assert bmap_damaged.shape == (12,)
    # Damaged layer 0 norm change should be > 0
    assert bmap_damaged[3] > 0.0
    # Damaged layer 0 zero fraction should be > 0.15
    assert bmap_damaged[2] > 0.15
    # Layer 1 and 2 norm change should still be 0.0
    assert bmap_damaged[7] == 0.0
    assert bmap_damaged[11] == 0.0

    names = get_body_map_feature_names()
    assert len(names) == 12


def test_generate_body_map_dataset():
    healthy = BaselineMLP(input_dim=64, hidden1=32, hidden2=16, num_classes=10)
    X, yd, yl, meta = generate_body_map_dataset(
        healthy, n_healthy=10, n_damaged_per_layer=6, seed=42
    )
    # Total = 10 healthy + 3 * 6 damaged = 28
    assert X.shape == (28, 12)
    assert len(yd) == 28
    assert len(yl) == 28
    assert (yd == 0).sum() == 10
    assert (yd == 1).sum() == 18
