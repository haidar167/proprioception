"""MLP architecture, training routines, and checkpoint utilities for MNIST."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import set_seed


class BaselineMLP(nn.Module):
    """Standard 3-layer MLP: 784 -> 256 -> 128 -> 10."""

    def __init__(self, input_dim: int = 784, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

        self.layers: List[nn.Linear] = [self.fc1, self.fc2, self.fc3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return logits and list of intermediate hidden activations for interoceptive monitoring."""
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        logits = self.fc3(h2)
        return logits, [h1, h2]

    def get_layer(self, index: int) -> nn.Linear:
        return self.layers[index]

    def num_layers(self) -> int:
        return len(self.layers)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_layer_parameters(self, layer_idx: int) -> int:
        layer = self.layers[layer_idx]
        return sum(p.numel() for p in layer.parameters())


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Evaluate model accuracy and average cross-entropy loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * len(y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def train_baseline_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int = 784,
    epochs: int = 3,
    lr: float = 1e-3,
    seed: int = 0,
    checkpoint_path: Union[str, Path, None] = None,
    device: str = "cpu",
) -> Tuple[BaselineMLP, Dict[str, List[float]]]:
    """Train baseline MLP on MNIST for specified epochs, reporting epoch metrics."""
    set_seed(seed)
    model = BaselineMLP(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    print(f"--- Training Baseline MLP ({input_dim}->256->128->10) for {epochs} epochs (seed={seed}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            total_samples += len(y)

        train_loss = running_loss / max(1, total_samples)
        test_loss, test_acc = evaluate_model(model, test_loader, device=device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"Healthy baseline checkpoint saved to {checkpoint_path}")

    return model, history
