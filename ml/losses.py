"""Losses. The classification loss is identical for both models."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .labels import NUM_CLASSES


def compute_class_weights(train_label_indices: list[int], mode: str) -> list[float]:
    """Weights per internal class index from TRAIN-split counts only (mean over train samples = 1)."""
    counts = np.bincount(np.asarray(train_label_indices), minlength=NUM_CLASSES).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"Training split has a class with no examples: {counts.tolist()}")
    if mode == "none":
        w = np.ones(NUM_CLASSES)
    elif mode == "balanced":
        w = counts.sum() / (NUM_CLASSES * counts)
    elif mode == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weighting {mode!r}")
    w = w / (w * counts).sum() * counts.sum()
    return [float(x) for x in w]


def soft_dice_loss(mask_logits: torch.Tensor, mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(mask_logits.float())
    inter = (p * mask).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + mask.sum() + smooth)


def segmentation_loss(mask_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(mask_logits.float(), mask)
    return bce + soft_dice_loss(mask_logits, mask)


class MultiTaskLoss:
    def __init__(self, class_weights: list[float], aux_weight: float, device: torch.device):
        self.weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.aux_weight = float(aux_weight)

    def __call__(self, out: dict, label_index: torch.Tensor, mask: torch.Tensor) -> dict:
        cls = F.cross_entropy(out["logits"].float(), label_index, weight=self.weight)
        seg = None
        total = cls
        if out["mask_logits"] is not None and self.aux_weight > 0:
            seg = segmentation_loss(out["mask_logits"], mask)
            total = cls + self.aux_weight * seg
        return {"total": total, "cls": cls, "seg": seg}
