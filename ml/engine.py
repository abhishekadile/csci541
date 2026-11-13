"""Shared train/eval loops used by training, evaluation, smoke test and benchmark."""
from __future__ import annotations

import contextlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from .labels import INDEX_TO_SOURCE, SOURCE_CLASS_IDS
from .losses import MultiTaskLoss
from .metrics import mask_overlap


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False. See README > Troubleshooting.")
    return dev


def autocast_ctx(device: torch.device, amp: bool):
    if device.type == "cuda" and amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def to_device(batch: dict, device: torch.device) -> dict:
    nb = device.type == "cuda"
    return {k: (v.to(device, non_blocking=nb) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_epoch(model, loader, crit: MultiTaskLoss, optimizer, scaler, device, amp: bool,
                grad_clip: float | None = None, max_batches: int | None = None, log_every: int = 0) -> dict:
    model.train()
    sums = {"total": 0.0, "cls": 0.0, "seg": 0.0}
    n_seg = n = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        b = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            out = model(b["image"])
        losses = crit(out, b["label_index"], b["mask"])
        scaler.scale(losses["total"]).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = b["image"].shape[0]
        n += bs
        sums["total"] += float(losses["total"].detach()) * bs
        sums["cls"] += float(losses["cls"].detach()) * bs
        if losses["seg"] is not None:
            sums["seg"] += float(losses["seg"].detach()) * bs
            n_seg += bs
        if log_every and (bi + 1) % log_every == 0:
            print(f"    batch {bi + 1}/{len(loader)} loss {sums['total'] / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return {"loss": sums["total"] / max(n, 1), "cls_loss": sums["cls"] / max(n, 1),
            "seg_loss": (sums["seg"] / n_seg) if n_seg else None, "n": n, "seconds": time.time() - t0}


@torch.no_grad()
def run_eval(model, loader, device, amp: bool, mask_threshold: float = 0.5, max_batches: int | None = None,
             want_mask_stats: bool = True) -> dict:
    """Forward pass over a loader. Returns predictions in ORIGINAL source class IDs."""
    model.eval()
    filenames: list[str] = []
    true_ids: list[int] = []
    pred_ids: list[int] = []
    probs: list[np.ndarray] = []
    ce_sum = 0.0
    mask_items: list[dict] = []
    t0 = time.time()
    index_to_source = [INDEX_TO_SOURCE[i] for i in range(len(SOURCE_CLASS_IDS))]
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        b = to_device(batch, device)
        with autocast_ctx(device, amp):
            out = model(b["image"])
        logits = out["logits"].float()
        ce_sum += float(F.cross_entropy(logits, b["label_index"], reduction="sum"))
        p = torch.softmax(logits, 1)
        idx = p.argmax(1).cpu().numpy()
        probs.append(p.cpu().numpy())
        pred_ids += [index_to_source[int(i)] for i in idx]  # back to source IDs 1..6
        true_ids += [int(s) for s in batch["source_id"]]
        filenames += list(batch["filename"])
        if want_mask_stats and out["mask_logits"] is not None:
            pm = (torch.sigmoid(out["mask_logits"].float()) > mask_threshold)
            gm = b["mask"] > 0.5
            for k in range(pm.shape[0]):
                mask_items.append(mask_overlap(pm[k, 0].cpu().numpy(), gm[k, 0].cpu().numpy()))
    n = max(len(true_ids), 1)
    return {"filenames": filenames, "true_ids": true_ids, "pred_ids": pred_ids,
            "probs": np.concatenate(probs) if probs else np.zeros((0, len(SOURCE_CLASS_IDS))),
            "ce_loss": ce_sum / n, "mask_items": mask_items or None, "seconds": time.time() - t0, "n": len(true_ids)}
