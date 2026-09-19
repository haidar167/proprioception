"""Evaluation metrics for damage detection, localization, and self-repair recovery."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute binary ROC-AUC score safely."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if len(np.unique(y_true_arr)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_arr, y_score_arr))


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], average: str = "macro"
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics including accuracy, precision, recall, and F1."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(
        precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0)
    )
    rec = float(recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def compute_recovery_percentage(
    acc_healthy: float, acc_damaged: float, acc_repaired: float
) -> float:
    """Compute percentage of accuracy lost to corruption that was successfully recovered.

    Recovery % = (acc_repaired - acc_damaged) / (acc_healthy - acc_damaged) * 100
    Bounded rationally if acc_healthy == acc_damaged.
    """
    damage_drop = acc_healthy - acc_damaged
    if abs(damage_drop) < 1e-6:
        return 100.0 if acc_repaired >= acc_healthy else 0.0
    recovery = (acc_repaired - acc_damaged) / damage_drop * 100.0
    return float(recovery)


def plot_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    filepath: Union[str, Path],
    title: str = "Damage Detection ROC Curve",
) -> float:
    """Plot and save ROC curve with dpi=150."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    auc = compute_auc(y_true_arr, y_score_arr)

    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return auc


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    filepath: Union[str, Path],
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix with counts and normalized percentages at dpi=150."""
    cm_arr = np.asarray(cm)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5.5, 4.5), dpi=150)
    plt.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks, class_names)

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            plt.text(
                j,
                i,
                f"{val}",
                horizontalalignment="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
