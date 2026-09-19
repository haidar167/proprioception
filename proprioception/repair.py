"""Self-repair routines: Targeted layer repair vs Full model repair on rehearsal buffer."""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .metrics import compute_recovery_percentage
from .models import BaselineMLP, evaluate_model


def run_repair_step_optimization(
    model: BaselineMLP,
    buffer_x: torch.Tensor,
    buffer_y: torch.Tensor,
    target_layer: Optional[int] = None,
    num_steps: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[BaselineMLP, float, int]:
    """Execute gradient fine-tuning steps on the rehearsal buffer.

    Args:
        model: Corrupted model.
        buffer_x: (N, D) rehearsal buffer inputs.
        buffer_y: (N,) rehearsal buffer labels.
        target_layer: If None, full-model repair. If int (0, 1, 2), freeze all other layers.
        num_steps: Number of gradient steps (default 500).
        batch_size: Mini-batch size sampled from buffer.
        lr: Learning rate.
        device: 'cpu'.
        seed: Random seed for mini-batch sampling.

    Returns:
        (repaired_model, wall_clock_seconds, num_params_touched)
    """
    repaired = copy.deepcopy(model).to(device)

    # Configure parameter freeze
    params_to_optimize = []
    num_params_touched = 0

    if target_layer is None:
        # Full model repair
        for p in repaired.parameters():
            p.requires_grad = True
            params_to_optimize.append(p)
            num_params_touched += p.numel()
    else:
        # Targeted layer repair: freeze all layers except target_layer
        for l_idx, layer in enumerate(repaired.layers):
            if l_idx == target_layer:
                for p in layer.parameters():
                    p.requires_grad = True
                    params_to_optimize.append(p)
                    num_params_touched += p.numel()
            else:
                for p in layer.parameters():
                    p.requires_grad = False

    optimizer = optim.Adam(params_to_optimize, lr=lr)
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(seed)
    buffer_size = len(buffer_x)

    repaired.train()
    start_time = time.perf_counter()

    for step in range(num_steps):
        # Sample mini-batch from rehearsal buffer
        batch_indices = torch.randint(0, buffer_size, (batch_size,), generator=generator)
        bx = buffer_x[batch_indices].to(device)
        by = buffer_y[batch_indices].to(device)

        optimizer.zero_grad()
        out = repaired(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start_time

    return repaired, elapsed, num_params_touched


def evaluate_repair_condition(
    condition_name: str,
    corrupted_model: BaselineMLP,
    test_loader: DataLoader,
    buffer_x: torch.Tensor,
    buffer_y: torch.Tensor,
    healthy_acc: float,
    corrupted_acc: float,
    target_layer: Optional[int],
    num_steps: int = 500,
    lr: float = 1e-3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate one repair condition (no repair, full repair, or targeted repair)."""
    if condition_name == "no_repair":
        return {
            "condition": "no_repair",
            "repaired_acc": corrupted_acc,
            "recovery_pct": 0.0,
            "repair_time_sec": 0.0,
            "params_touched": 0,
            "num_steps": 0,
        }

    repaired_model, elapsed, touched = run_repair_step_optimization(
        corrupted_model,
        buffer_x,
        buffer_y,
        target_layer=target_layer,
        num_steps=num_steps,
        lr=lr,
        seed=seed,
    )
    _, repaired_acc = evaluate_model(repaired_model, test_loader)
    recovery_pct = compute_recovery_percentage(healthy_acc, corrupted_acc, repaired_acc)

    return {
        "condition": condition_name,
        "repaired_acc": float(repaired_acc),
        "recovery_pct": float(recovery_pct),
        "repair_time_sec": float(elapsed),
        "params_touched": int(touched),
        "num_steps": num_steps,
    }
