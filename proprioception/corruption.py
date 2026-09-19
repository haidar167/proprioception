"""Weight corruption generators: zero-out mask and Gaussian noise."""

import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .models import BaselineMLP


def clone_model(model: BaselineMLP) -> BaselineMLP:
    """Create a deep copy of a BaselineMLP model with identical parameters."""
    return copy.deepcopy(model)


def apply_zero_corruption(
    model: BaselineMLP,
    layer_idx: int,
    fraction: float,
    seed: Optional[int] = None,
) -> BaselineMLP:
    """Zero out a random fraction f of weights in layer_idx.

    Args:
        model: BaselineMLP instance (modified in-place or cloned).
        layer_idx: Layer index (0, 1, or 2).
        fraction: Fraction of weights to zero out in (0.0, 1.0].
        seed: Optional random seed for reproducible mask.

    Returns:
        The damaged model.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"Fraction must be in [0.0, 1.0], got {fraction}")

    layer = model.get_layer(layer_idx)
    weight = layer.weight.data

    if seed is not None:
        torch.manual_seed(seed)

    # Generate random binary mask
    mask = (torch.rand_like(weight) >= fraction).float()
    weight.mul_(mask)
    return model


def apply_gaussian_corruption(
    model: BaselineMLP,
    layer_idx: int,
    noise_std: float,
    seed: Optional[int] = None,
) -> BaselineMLP:
    """Add zero-mean Gaussian noise N(0, noise_std^2) to weights in layer_idx.

    Args:
        model: BaselineMLP instance.
        layer_idx: Layer index (0, 1, or 2).
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Optional random seed for reproducible noise.

    Returns:
        The damaged model.
    """
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    layer = model.get_layer(layer_idx)
    weight = layer.weight.data

    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(weight) * noise_std
    weight.add_(noise)
    return model


def corrupt_model(
    healthy_model: BaselineMLP,
    damage_type: str,
    layer_idx: int,
    intensity: float,
    seed: Optional[int] = None,
) -> BaselineMLP:
    """Clone healthy model and apply specified corruption type to layer_idx.

    Args:
        healthy_model: Base model.
        damage_type: 'zero' or 'gaussian'.
        layer_idx: Layer index to corrupt (0, 1, or 2).
        intensity: fraction for zero-out, or noise_std for gaussian.
        seed: Random seed.

    Returns:
        Damaged model clone.
    """
    damaged = clone_model(healthy_model)
    if damage_type == "zero":
        return apply_zero_corruption(damaged, layer_idx, fraction=intensity, seed=seed)
    elif damage_type == "gaussian":
        return apply_gaussian_corruption(damaged, layer_idx, noise_std=intensity, seed=seed)
    else:
        raise ValueError(f"Unknown damage type: {damage_type}")
