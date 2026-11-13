"""Datasets and batching at native resolution.

Nothing here resizes or center-crops: tensors keep the file's HxW (verified 480x480).
If sizes ever differ inside a batch, `pad_collate` zero-pads to the largest HxW in the
batch (after normalization, so padding equals the mean intensity) and records the
valid extent; with the verified uniform data this is a no-op.
Geometric augmentation is applied identically to image and mask; masks are only
moved by integer flips / shifts so no interpolation is involved (nearest by construction).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_index import Record, load_image_u8, load_mask_binary
from .labels import encode


def normalize_u8(img_u8: np.ndarray, mean: float, std: float) -> torch.Tensor:
    x = torch.from_numpy(img_u8.astype(np.float32) / 255.0)
    return ((x - mean) / std).unsqueeze(0)


def _shift(arr: np.ndarray, dy: int, dx: int, mode: str) -> np.ndarray:
    if dy == 0 and dx == 0:
        return arr
    h, w = arr.shape
    p = max(abs(dy), abs(dx))
    padded = np.pad(arr, p, mode=mode)
    return padded[p - dy : p - dy + h, p - dx : p - dx + w]


def augment_pair(img: np.ndarray, mask: np.ndarray | None, cfg: dict, rng: np.random.Generator):
    """img float32 [0,1] HxW, mask uint8 {0,1} HxW or None. Returns augmented copies."""
    if rng.random() < cfg.get("vflip_p", 0.0):
        img = img[::-1]
        mask = mask[::-1] if mask is not None else None
    if rng.random() < cfg.get("hflip_p", 0.0):
        img = img[:, ::-1]
        mask = mask[:, ::-1] if mask is not None else None
    t = int(cfg.get("max_translate_px", 0))
    if t > 0:
        dy, dx = int(rng.integers(-t, t + 1)), int(rng.integers(-t, t + 1))
        img = _shift(np.ascontiguousarray(img), dy, dx, "edge")
        mask = _shift(np.ascontiguousarray(mask), dy, dx, "constant") if mask is not None else None
    b, c = cfg.get("brightness", 0.0), cfg.get("contrast", 0.0)
    if b or c:
        img = (img - 0.5) * (1.0 + rng.uniform(-c, c)) + 0.5 + rng.uniform(-b, b)
        img = np.clip(img, 0.0, 1.0)
    return np.ascontiguousarray(img, dtype=np.float32), (np.ascontiguousarray(mask) if mask is not None else None)


class WaferDataset(Dataset):
    def __init__(self, records: list[Record], mean: float, std: float, with_masks: bool = True,
                 augment: dict | None = None):
        self.records, self.mean, self.std = records, mean, std
        self.with_masks, self.augment = with_masks, augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        img = load_image_u8(r.image_path)
        mask = load_mask_binary(r.mask_path) if (self.with_masks and r.mask_path is not None) else None
        if self.augment is not None:
            # seeded through torch's RNG, so DataLoader/worker seeding controls reproducibility
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            f, mask = augment_pair(img.astype(np.float32) / 255.0, mask, self.augment, rng)
            x = ((torch.from_numpy(f) - self.mean) / self.std).unsqueeze(0)
        else:
            x = normalize_u8(img, self.mean, self.std)
        m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0) if mask is not None else torch.zeros_like(x)
        return {"image": x, "mask": m, "has_mask": mask is not None, "label_index": encode(r.label),
                "source_id": r.label, "filename": r.filename}


def pad_collate(batch: list[dict]) -> dict:
    hs = [b["image"].shape[1] for b in batch]
    ws = [b["image"].shape[2] for b in batch]
    H, W = max(hs), max(ws)

    def pad(t: torch.Tensor) -> torch.Tensor:
        return t if t.shape[1:] == (H, W) else torch.nn.functional.pad(t, (0, W - t.shape[2], 0, H - t.shape[1]))

    return {
        "image": torch.stack([pad(b["image"]) for b in batch]),
        "mask": torch.stack([pad(b["mask"]) for b in batch]),
        "has_mask": torch.tensor([b["has_mask"] for b in batch]),
        "label_index": torch.tensor([b["label_index"] for b in batch], dtype=torch.long),
        "source_id": torch.tensor([b["source_id"] for b in batch], dtype=torch.long),
        "filename": [b["filename"] for b in batch],
        "valid_hw": torch.tensor(list(zip(hs, ws))),
    }


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int,
                device_is_cuda: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    kw = {}
    if num_workers > 0:
        kw = {"persistent_workers": True, "prefetch_factor": 2}
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=g,
                      pin_memory=device_is_cuda, collate_fn=pad_collate, drop_last=False, **kw)


def records_for_split(manifest: list[dict], by_name: dict[str, Record], split: str) -> list[Record]:
    return [by_name[r["filename"]] for r in manifest if r["split"] == split]


def parse_hw(cfg: dict) -> tuple[int, int]:
    h, w = cfg["data"]["input_size"]
    return int(h), int(w)
