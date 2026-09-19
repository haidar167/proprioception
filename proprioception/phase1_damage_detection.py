"""Phase 1: Baseline training, body map extraction, damage detection, and localization."""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from .bodymap import extract_body_map, get_body_map_feature_names
from .corruption import corrupt_model
from .data import load_mnist_data
from .metrics import compute_auc, compute_classification_metrics, plot_confusion_matrix, plot_roc_curve
from .models import BaselineMLP, evaluate_model, train_baseline_model
from .utils import save_results, set_seed


def generate_body_map_dataset(
    healthy_model: BaselineMLP,
    n_healthy: int = 200,
    n_damaged_per_layer: int = 80,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Generate a diverse dataset of healthy and damaged model body maps.

    Args:
        healthy_model: Baseline trained model.
        n_healthy: Number of healthy model variations.
        n_damaged_per_layer: Number of damaged variations for each layer (split across zero & gaussian).
        seed: Random seed.

    Returns:
        X: (N, 12) feature matrix of body maps.
        y_detect: (N,) binary labels (0=healthy, 1=damaged).
        y_local: (N,) layer labels (-1 for healthy, 0, 1, 2 for damaged layer).
        metadata: List of dicts with damage specifics.
    """
    rng = np.random.RandomState(seed)
    body_maps = []
    labels_detect = []
    labels_local = []
    metadata = []

    # 1. Healthy models (pure healthy + tiny numerical jitters)
    for i in range(n_healthy):
        m_copy = copy.deepcopy(healthy_model)
        # Add tiny non-damaging floating point jitter in 50% of healthy copies
        if i % 2 == 1:
            for p in m_copy.parameters():
                jitter = torch.randn_like(p) * 1e-6
                p.data.add_(jitter)

        bmap = extract_body_map(m_copy, healthy_model)
        body_maps.append(bmap)
        labels_detect.append(0)
        labels_local.append(-1)
        metadata.append({"type": "healthy", "layer": -1, "intensity": 0.0})

    # 2. Damaged models across 3 layers
    for layer_idx in range(3):
        # Half zero-out corruption, half gaussian corruption
        n_zero = n_damaged_per_layer // 2
        n_gauss = n_damaged_per_layer - n_zero

        # Zero corruptions: fraction f in [0.03, 0.40]
        for _ in range(n_zero):
            f = float(rng.uniform(0.03, 0.40))
            sub_seed = int(rng.randint(0, 1000000))
            damaged = corrupt_model(healthy_model, "zero", layer_idx, intensity=f, seed=sub_seed)
            bmap = extract_body_map(damaged, healthy_model)
            body_maps.append(bmap)
            labels_detect.append(1)
            labels_local.append(layer_idx)
            metadata.append({"type": "zero", "layer": layer_idx, "intensity": f})

        # Gaussian corruptions: std in [0.01, 0.25]
        for _ in range(n_gauss):
            std = float(rng.uniform(0.01, 0.25))
            sub_seed = int(rng.randint(0, 1000000))
            damaged = corrupt_model(healthy_model, "gaussian", layer_idx, intensity=std, seed=sub_seed)
            bmap = extract_body_map(damaged, healthy_model)
            body_maps.append(bmap)
            labels_detect.append(1)
            labels_local.append(layer_idx)
            metadata.append({"type": "gaussian", "layer": layer_idx, "intensity": std})

    X = np.array(body_maps, dtype=np.float32)
    y_detect = np.array(labels_detect, dtype=np.int64)
    y_local = np.array(labels_local, dtype=np.int64)

    return X, y_detect, y_local, metadata


def run_phase1(
    seed: int = 0,
    epochs: int = 3,
    output_dir: str = "results",
    figure_dir: str = "figures",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Execute complete Phase 1 pipeline."""
    set_seed(seed)
    print("=" * 70)
    print("PHASE 1: PROPRIOCEPTION — DAMAGE DETECTION & LOCALIZATION")
    print("=" * 70)

    # 1. Load MNIST data
    train_loader, test_loader, train_dataset, test_dataset, input_dim = load_mnist_data(batch_size=128)

    # 2. Train or load baseline healthy model
    ckpt_path = Path(checkpoint_dir) / "baseline_healthy.pt"
    healthy_model = BaselineMLP(input_dim=input_dim)
    if ckpt_path.exists():
        print(f"Loading existing baseline model from {ckpt_path}")
        healthy_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        _, test_acc = evaluate_model(healthy_model, test_loader)
        print(f"Loaded model test accuracy: {test_acc*100:.2f}%")
    else:
        healthy_model, history = train_baseline_model(
            train_loader, test_loader, input_dim=input_dim, epochs=epochs, seed=seed, checkpoint_path=ckpt_path
        )
        test_acc = history["test_acc"][-1]

    # 3. Generate dataset of body maps
    print("\nGenerating body maps for healthy and corrupted models...")
    X, y_detect, y_local, meta = generate_body_map_dataset(
        healthy_model, n_healthy=200, n_damaged_per_layer=80, seed=seed
    )
    print(f"Dataset generated: Total models = {len(X)} | Healthy = {(y_detect == 0).sum()} | Damaged = {(y_detect == 1).sum()}")

    # 4. Train/Test split for Damage Detection
    X_train, X_test, yd_train, yd_test, yl_train, yl_test = train_test_split(
        X, y_detect, y_local, test_size=0.30, random_state=seed, stratify=y_detect
    )

    # 5. Damage Detection (Binary Classifier)
    # Using LogisticRegression with L2 regularization
    detector = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000))
    detector.fit(X_train, yd_train)
    yd_pred = detector.predict(X_test)
    yd_scores = detector.predict_proba(X_test)[:, 1]

    auc_score = compute_auc(yd_test, yd_scores)
    detect_metrics = compute_classification_metrics(yd_test, yd_pred)
    detect_metrics["auc"] = auc_score

    print("\n--- Damage Detection (Binary) ---")
    print(f"ROC-AUC: {auc_score:.4f} (Requirement: > 0.95)")
    print(f"Accuracy: {detect_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {detect_metrics['precision']:.4f} | Recall: {detect_metrics['recall']:.4f} | F1: {detect_metrics['f1']:.4f}")

    # 6. Damage Localization (3-Class Classifier on damaged models only)
    damaged_train_idx = np.where(yd_train == 1)[0]
    damaged_test_idx = np.where(yd_test == 1)[0]

    X_train_dam = X_train[damaged_train_idx]
    yl_train_dam = yl_train[damaged_train_idx]
    X_test_dam = X_test[damaged_test_idx]
    yl_test_dam = yl_test[damaged_test_idx]

    localizer = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed)
    )
    localizer.fit(X_train_dam, yl_train_dam)
    yl_pred = localizer.predict(X_test_dam)

    local_metrics = compute_classification_metrics(yl_test_dam, yl_pred, average="macro")
    loc_acc = local_metrics["accuracy"]

    print("\n--- Damage Localization (3-Class: Layer 0, 1, 2) ---")
    print(f"Localization Accuracy: {loc_acc*100:.2f}% (Requirement: > 80%)")
    print(f"Macro F1: {local_metrics['f1']:.4f} | Precision: {local_metrics['precision']:.4f} | Recall: {local_metrics['recall']:.4f}")
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    for row in local_metrics["confusion_matrix"]:
        print(f"  {row}")

    # 7. Verification of acceptance criteria
    auc_passed = auc_score > 0.95
    loc_passed = loc_acc > 0.80
    print("\n--- Acceptance Check ---")
    print(f"AUC > 0.95: {'PASSED' if auc_passed else 'FAILED'} ({auc_score:.4f})")
    print(f"Localization Accuracy > 80%: {'PASSED' if loc_passed else 'FAILED'} ({loc_acc*100:.2f}%)")

    # 8. Visualizations (dpi=150)
    fig_dir = Path(figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "phase1_detection_localization.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    # Subplot 1: ROC Curve
    fpr, tpr, _ = roc_curve(yd_test, yd_scores)
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Body Map Detector (AUC = {auc_score:.4f})")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5)
    axes[0].set_title("Damage Detection ROC Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Localization Confusion Matrix
    cm = np.array(local_metrics["confusion_matrix"])
    im = axes[1].imshow(cm, cmap="Blues", interpolation="nearest")
    axes[1].set_title(f"Damage Localization (Acc = {loc_acc*100:.1f}%)", fontsize=12, fontweight="bold")
    class_labels = ["Layer 0 (fc1)", "Layer 1 (fc2)", "Layer 2 (fc3)"]
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(class_labels, rotation=15, ha="right", fontsize=9)
    axes[1].set_yticklabels(class_labels, fontsize=9)
    axes[1].set_xlabel("Predicted Layer", fontsize=10)
    axes[1].set_ylabel("True Damaged Layer", fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            val = cm[r, c]
            axes[1].text(c, r, f"{val}", ha="center", va="center",
                         color="white" if val > thresh else "black", fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure: {fig_path}")

    # 9. Save JSON results
    res_path = Path(output_dir) / "phase1.json"
    results_payload = {
        "healthy_baseline_test_acc": float(test_acc),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "detection_metrics": detect_metrics,
        "localization_metrics": local_metrics,
        "acceptance": {
            "auc_threshold": 0.95,
            "auc_achieved": float(auc_score),
            "auc_passed": bool(auc_passed),
            "localization_threshold": 0.80,
            "localization_achieved": float(loc_acc),
            "localization_passed": bool(loc_passed),
            "overall_passed": bool(auc_passed and loc_passed),
        },
        "feature_names": get_body_map_feature_names(),
    }
    saved_data = save_results(results_payload, res_path, seed=seed)
    print(f"Saved results: {res_path}")

    return saved_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    run_phase1(seed=args.seed, epochs=args.epochs)
