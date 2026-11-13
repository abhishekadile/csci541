"""Short hardware benchmark at full 480x480 resolution.

    python -m ml.benchmark [--batch-size 8] [--train-batches 30] [--val-batches 15]

Times real training and validation batches (data loading included), reports peak GPU memory and
estimates a full run from measured throughput. The estimate is an UPPER bound when early
stopping triggers. Nothing trained here is kept: results go to artifacts/smoke/benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .config import artifacts_dir, load_config
from .engine import pick_device, run_eval, train_epoch
from .losses import MultiTaskLoss
from .models import build_model, count_parameters
from .train import build_context, seed_everything


def bench_model(cfg_name: str, device, batch_size: int, n_train: int, n_val: int,
                num_workers: int | None = None) -> dict:
    cfg = load_config(cfg_name)
    cfg["train"]["batch_size"] = batch_size
    if num_workers is not None:
        cfg["data"]["num_workers"] = num_workers
    seed_everything(cfg["seed"])
    ctx = build_context(cfg, device, batch_size)
    model = build_model(cfg["model"]).to(device)
    crit = MultiTaskLoss(ctx.class_weights, cfg["loss"]["aux_weight"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=ctx.amp)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_epoch(model, ctx.train_loader, crit, opt, scaler, device, ctx.amp, max_batches=3)  # warm-up
    run_eval(model, ctx.val_loader, device, ctx.amp, want_mask_stats=False, max_batches=2)  # eval warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr = train_epoch(model, ctx.train_loader, crit, opt, scaler, device, ctx.amp, cfg["train"]["grad_clip_norm"],
                     max_batches=n_train)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.time() - t0
    peak_train = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    t0 = time.time()
    ev = run_eval(model, ctx.val_loader, device, ctx.amp, want_mask_stats=False, max_batches=n_val)
    if device.type == "cuda":
        torch.cuda.synchronize()
    val_s = time.time() - t0
    peak_all = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    train_ips, val_ips = tr["n"] / train_s, ev["n"] / val_s
    n_tr, n_va = len(ctx.train_ds), len(ctx.val_ds)
    epoch_s = n_tr / train_ips + n_va / val_ips
    max_ep = cfg["train"]["max_epochs"]
    return {
        "model": cfg["name"], "parameters": count_parameters(model), "batch_size": batch_size,
        "amp": ctx.amp, "train_images_per_s": train_ips, "val_images_per_s": val_ips,
        "train_images_timed": tr["n"], "val_images_timed": ev["n"],
        "peak_gpu_mb_train": peak_train, "peak_gpu_mb_train_and_val": peak_all,
        "estimated_epoch_seconds": epoch_s, "max_epochs": max_ep,
        "estimated_full_run_minutes_at_max_epochs": epoch_s * max_ep / 60.0,
        "train_set_size": n_tr, "val_set_size": n_va,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-batches", type=int, default=30)
    ap.add_argument("--val-batches", type=int, default=15)
    ap.add_argument("--num-workers", type=int, help="DataLoader workers (default: value in configs/common.yaml)")
    a = ap.parse_args()
    device = pick_device(a.device)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none (CPU)"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if device.type == "cuda" else None
    print(f"Benchmark device={device} gpu={gpu} batch={a.batch_size} resolution=480x480 (native)")
    results = []
    for name in ("model_a.yaml", "model_b.yaml"):
        r = bench_model(name, device, a.batch_size, a.train_batches, a.val_batches, a.num_workers)
        results.append(r)
        mem = f"{r['peak_gpu_mb_train_and_val']:.0f} MB peak GPU" if r["peak_gpu_mb_train_and_val"] else "CPU"
        print(f"  {r['model']}: train {r['train_images_per_s']:.1f} img/s, val {r['val_images_per_s']:.1f} img/s, "
              f"{mem}; est. epoch {r['estimated_epoch_seconds']:.0f}s -> "
              f"{r['estimated_full_run_minutes_at_max_epochs']:.0f} min for {r['max_epochs']} epochs (upper bound)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total = sum(r["estimated_full_run_minutes_at_max_epochs"] for r in results)
    print(f"Estimated both models, {results[0]['max_epochs']} epochs each, no early stop: {total:.0f} min ({total / 60:.1f} h)")
    out = artifacts_dir() / "smoke" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": True, "not_a_model": True, "device": str(device), "gpu": gpu,
                               "gpu_total_gb": total_gb, "results": results,
                               "estimated_total_minutes_both_models": total}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
