"""Body Map Extractor: internal weight statistics per layer."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .models import BaselineMLP


def extract_layer_statistics(
    current_weight: torch.Tensor,
    healthy_weight: torch.Tensor,
    zero_threshold: float = 1e-9,
) -> Dict[str, float]:
    """Compute the 4 canonical proprioceptive statistics for a single weight matrix:
    1. Weight L2 norm: ||W||_2
    2. Mean absolute weight: E[|w|]
    3. Fraction of exact/near zeros: E[|w| < threshold]
    4. Norm change vs healthy checkpoint: ||W - W_healthy||_2
    """
    with torch.no_grad():
        w_curr = current_weight.detach().float()
        w_ref = healthy_weight.detach().float()

        l2_norm = float(torch.norm(w_curr, p=2).item())
        mean_abs = float(torch.mean(torch.abs(w_curr)).item())
        zero_frac = float((torch.abs(w_curr) <= zero_threshold).float().mean().item())
        delta_norm = float(torch.norm(w_curr - w_ref, p=2).item())

    return {
        "l2_norm": l2_norm,
        "mean_abs": mean_abs,
        "zero_frac": zero_frac,
        "norm_change": delta_norm,
    }


def extract_body_map(
    model: BaselineMLP,
    healthy_reference: BaselineMLP,
    zero_threshold: float = 1e-9,
) -> np.ndarray:
    """Extract 12-dimensional body map feature vector from 3 layers of BaselineMLP.

    Returns:
        1D numpy array of shape (12,) with dtype float32.
    """
    features: List[float] = []
    for l_idx in range(model.num_layers()):
        w_curr = model.get_layer(l_idx).weight
        w_ref = healthy_reference.get_layer(l_idx).weight
        stats = extract_layer_statistics(w_curr, w_ref, zero_threshold=zero_threshold)
        features.extend([
            stats["l2_norm"],
            stats["mean_abs"],
            stats["zero_frac"],
            stats["norm_change"],
        ])
    return np.array(features, dtype=np.float32)


def get_body_map_feature_names() -> List[str]:
    """Return human-readable names for the 12 body map features."""
    names = []
    for l_idx in range(3):
        names.extend([
            f"layer{l_idx}_l2_norm",
            f"layer{l_idx}_mean_abs",
            f"layer{l_idx}_zero_frac",
            f"layer{l_idx}_norm_change",
        ])
    return names
