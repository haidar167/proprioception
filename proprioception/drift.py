"""Interoception vs Proprioception evaluation under covariate drift (Permuted-MNIST)."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import BaselineMLP, evaluate_model


def compute_interoceptive_signals(
    model: BaselineMLP,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: int = 10,
) -> Dict[str, float]:
    """Compute activation and prediction statistics reflecting internal functional state (interoception).

    Returns:
        mean_entropy: Mean Shannon entropy of softmax predictions.
        mean_max_confidence: Mean maximum softmax probability.
        mean_h1_norm: Average L2 norm of Layer 1 hidden activations.
        mean_h2_norm: Average L2 norm of Layer 2 hidden activations.
        accuracy: Classification accuracy.
    """
    model.eval()
    entropies = []
    max_confs = []
    h1_norms = []
    h2_norms = []
    correct = 0
    total = 0

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(dataloader):
            if b_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits, (h1, h2) = model.forward_with_activations(x)
            probs = F.softmax(logits, dim=1)

            # Shannon entropy: -sum(p * log(p + 1e-12))
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            entropies.extend(entropy.cpu().numpy().tolist())

            max_p, preds = torch.max(probs, dim=1)
            max_confs.extend(max_p.cpu().numpy().tolist())
            correct += (preds == y).sum().item()
            total += len(y)

            h1_norms.append(torch.norm(h1, p=2, dim=1).mean().item())
            h2_norms.append(torch.norm(h2, p=2, dim=1).mean().item())

    return {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_max_confidence": float(np.mean(max_confs)) if max_confs else 0.0,
        "mean_h1_norm": float(np.mean(h1_norms)) if h1_norms else 0.0,
        "mean_h2_norm": float(np.mean(h2_norms)) if h2_norms else 0.0,
        "accuracy": float(correct / max(1, total)),
    }
