"""Data utilities for MNIST, rehearsal buffering, and permuted-MNIST drift."""

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms


def get_default_data_dir() -> Path:
    """Return default data directory path."""
    return Path(__file__).resolve().parent.parent / "data"


def load_mnist_data(
    batch_size: int = 128,
    data_dir: Union[str, Path, None] = None,
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, int]:
    """Load MNIST dataset, falling back to sklearn digits if download/read fails.

    Returns:
        (train_loader, test_loader, train_dataset, test_dataset, input_dim)
    """
    target_dir = Path(data_dir) if data_dir else get_default_data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=str(target_dir), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(target_dir), train=False, download=True, transform=transform
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        return train_loader, test_loader, train_dataset, test_dataset, 784
    except Exception as e:
        print(f"MNIST download/load fallback to sklearn digits: {e}")
        digits = load_digits()
        x_norm = digits.data / 16.0
        y_targets = digits.target
        x_train, x_test, y_train, y_test = train_test_split(
            x_norm, y_targets, test_size=0.2, random_state=42, stratify=y_targets
        )
        train_dataset = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).long(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).long(),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, train_dataset, test_dataset, 64


def create_rehearsal_buffer(
    train_dataset: Dataset, buffer_size: int = 500, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a deterministic rehearsal buffer of stored (input, label) pairs from training data."""
    rng = np.random.RandomState(seed)
    total = len(train_dataset)
    indices = rng.choice(total, size=min(buffer_size, total), replace=False)

    inputs, targets = [], []
    for idx in indices:
        item = train_dataset[int(idx)]
        x, y = item[0], item[1]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        inputs.append(x.unsqueeze(0))
        targets.append(y.unsqueeze(0))

    buffer_x = torch.cat(inputs, dim=0)
    buffer_y = torch.cat(targets, dim=0)
    return buffer_x, buffer_y


class PermutedDataset(Dataset):
    """Wraps an existing dataset and applies a fixed pixel permutation to inputs."""

    def __init__(self, base_dataset: Dataset, perm_indices: torch.Tensor):
        self.base_dataset = base_dataset
        self.perm_indices = perm_indices

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        x, y = self.base_dataset[idx]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        flat_x = torch.flatten(x)
        permuted_x = flat_x[self.perm_indices]
        return permuted_x, y


def get_permuted_loader(
    test_dataset: Dataset,
    permutation_seed: int,
    input_dim: int = 784,
    batch_size: int = 128,
) -> DataLoader:
    """Create a test DataLoader with a deterministic pixel permutation."""
    rng = np.random.RandomState(permutation_seed)
    perm_indices = torch.from_numpy(rng.permutation(input_dim)).long()
    perm_dataset = PermutedDataset(test_dataset, perm_indices)
    return DataLoader(perm_dataset, batch_size=batch_size, shuffle=False)
