"""
Unified MedMNIST v2 (224x224) DataLoader with:
  - Automatic dataset downloading via `medmnist`
  - Optional stratified subsampling (for low-budget PathMNIST experiments)
  - Inverse-frequency class weights for imbalanced clinical classification
  - Zero-shot test-time corruption transforms (Gaussian noise & Gaussian blur)
"""

import os
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import medmnist
from medmnist import INFO


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AddGaussianNoise:
    """Adds zero-mean Gaussian noise in normalized [0,1] pixel space before normalization."""

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noisy = tensor + torch.randn_like(tensor) * self.std
        return torch.clamp(noisy, 0.0, 1.0)


def get_transforms(is_train: bool, corruption: str = "clean", severity: float = 0.0):
    ops = [T.Lambda(lambda img: img.convert("RGB"))]
    if is_train:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
        ]
    else:
        if corruption == "gaussian_blur" and severity > 0:
            k = int(severity) if int(severity) % 2 == 1 else int(severity) + 1
            ops.append(T.GaussianBlur(kernel_size=max(3, k), sigma=(1.0, 2.0)))

    ops.append(T.ToTensor())

    if not is_train and corruption == "gaussian_noise" and severity > 0:
        ops.append(AddGaussianNoise(std=severity))

    ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)


def build_medmnist_loaders(
    dataset_name: str = "dermamnist",
    batch_size: int = 32,
    num_workers: int = 4,
    data_root: str = "./data",
    train_fraction: float = 1.0,
    seed: int = 42,
    test_corruption: str = "clean",
    corruption_severity: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, torch.Tensor]:
    """
    Returns (train_loader, val_loader, test_loader, num_classes, class_weights).
    """
    os.makedirs(data_root, exist_ok=True)
    name = dataset_name.lower()
    if name not in INFO:
        raise ValueError(f"Dataset '{name}' not in MedMNIST v2 INFO.")

    info = INFO[name]
    num_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    train_tf = get_transforms(is_train=True)
    eval_tf = get_transforms(is_train=False, corruption="clean")
    test_tf = get_transforms(is_train=False, corruption=test_corruption, severity=corruption_severity)

    train_ds = DataClass(split="train", transform=train_tf, download=True, size=224, root=data_root)
    val_ds = DataClass(split="val", transform=eval_tf, download=True, size=224, root=data_root)
    test_ds = DataClass(split="test", transform=test_tf, download=True, size=224, root=data_root)

    labels = np.array(train_ds.labels).reshape(-1).astype(int)

    # Stratified subsampling (e.g., train_fraction=0.2 for fast PathMNIST regime)
    if 0.0 < train_fraction < 1.0:
        rng = np.random.default_rng(seed)
        selected_indices = []
        for c in range(num_classes):
            c_idx = np.where(labels == c)[0]
            k = max(1, int(round(len(c_idx) * train_fraction)))
            selected_indices.extend(rng.choice(c_idx, size=k, replace=False).tolist())
        train_ds = Subset(train_ds, selected_indices)
        labels = labels[selected_indices]

    # Inverse-frequency class weights normalized to mean 1.0
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = len(labels) / (num_classes * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader, num_classes, class_weights
