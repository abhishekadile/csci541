"""Train one model (A or B) from scratch.

    python -m ml.train model_a.yaml            # baseline
    python -m ml.train model_b.yaml            # mask-supervised
    python -m ml.train model_b.yaml --resume   # continue an interrupted run
    python -m ml.train model_a.yaml --batch-size 4   # out-of-memory? keep resolution, lower batch size

Both models share the split, normalization, class weights, classification loss, augmentation,
optimizer, epoch budget and model-selection rule (best validation macro-F1, ties -> lower
validation cross-entropy). Test data is never touched here.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoint import build_checkpoint, save_checkpoint
from .config import get_data_root, load_config, run_dir, split_dir, weights_dir
from .data import WaferDataset, make_loader, parse_hw, records_for_split
from .dataset_index import read_records
from .engine import pick_device, run_eval, train_epoch
from .labels import encode
from .losses import MultiTaskLoss, compute_class_weights
from .metrics import classification_report, confusion_matrix, inspection_report
from .models import build_model, count_parameters
from .split import load_manifest


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Context:
    cfg: dict
    device: torch.device
    amp: bool
    norm: dict
    class_weights: list[float]
    train_ds: WaferDataset
    val_ds: WaferDataset
    train_loader: object
    val_loader: object
    train_labels: list[int]


def build_context(cfg: dict, device: torch.device, batch_size: int | None = None) -> Context:
    manifest = load_manifest(cfg)
    records = read_records(get_data_root(), require_masks=True)
    by_name = {r.filename: r for r in records}
    norm = json.loads((split_dir(cfg) / "normalization.json").read_text())
    tr = records_for_split(manifest, by_name, "train")
    va = records_for_split(manifest, by_name, "val")
    tcfg = cfg["train"]
    amp = bool(tcfg["amp"]) and device.type == "cuda"
    weights = compute_class_weights([encode(r.label) for r in tr], tcfg["class_weighting"])
    with_masks = bool(cfg["model"]["with_decoder"])
    train_ds = WaferDataset(tr, norm["mean"], norm["std"], with_masks=with_masks, augment=cfg["data"]["augment"])
    val_ds = WaferDataset(va, norm["mean"], norm["std"], with_masks=True, augment=None)
    bs = batch_size or tcfg["batch_size"]
    nw = cfg["data"]["num_workers"]
    return Context(cfg, device, amp, norm, weights, train_ds, val_ds,
                   make_loader(train_ds, bs, True, nw, cfg["seed"], device.type == "cuda"),
                   make_loader(val_ds, bs, False, nw, cfg["seed"], device.type == "cuda"),
                   [r.label for r in tr])


def _is_better(f1: float, ce: float, best: dict) -> bool:
    if best["f1"] is None:
        return True
    return f1 > best["f1"] + 1e-9 or (abs(f1 - best["f1"]) <= 1e-9 and ce < best["ce"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="model_a.yaml or model_b.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="discard an existing run directory content")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, help="DataLoader worker processes (default from configs/common.yaml)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    device = pick_device(args.device)
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = device.type == "cuda"

    out = run_dir(cfg)          # logs / summaries
    wdir = weights_dir(cfg)     # best.pt, latest.pt, resume_state.pt
    resume_path = wdir / "resume_state.pt"
    if (wdir / "latest.pt").exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"{wdir} already has a run. Use --resume to continue or --overwrite to start over.")
    if args.resume and not resume_path.exists():
        raise SystemExit(f"Nothing to resume: {resume_path} not found.")
    out.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(cfg, device)
    tcfg, mcfg = cfg["train"], cfg["model"]
    model = build_model(mcfg).to(device)
    crit = MultiTaskLoss(ctx.class_weights, cfg["loss"]["aux_weight"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=ctx.amp)

    print(f"[{cfg['name']}] device={device} amp={ctx.amp} batch={ctx.train_loader.batch_size} "
          f"params={count_parameters(model):,} train={len(ctx.train_ds)} val={len(ctx.val_ds)}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  class weights (train split only, {tcfg['class_weighting']}): "
          f"{[round(w, 3) for w in ctx.class_weights]}   aux_weight={cfg['loss']['aux_weight']}")

    start_epoch, bad_epochs = 1, 0
    best = {"f1": None, "ce": None, "epoch": None}
    history: list[dict] = []
    if args.resume:
        st = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, bad_epochs, best = st["epoch"] + 1, st["bad_epochs"], st["best"]
        history = json.loads(st["history_json"])
        print(f"  resumed after epoch {st['epoch']} (best val macro-F1 {best['f1']} at epoch {best['epoch']})")
    elif args.overwrite:
        for f in ("log.csv", "history.json", "run_summary.json"):
            (out / f).unlink(missing_ok=True)
        for f in ("best.pt", "latest.pt", "resume_state.pt"):
            (wdir / f).unlink(missing_ok=True)

    training_meta = {
        "seed": cfg["seed"], "batch_size": ctx.train_loader.batch_size, "max_epochs": tcfg["max_epochs"],
        "lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"], "optimizer": tcfg["optimizer"], "amp": ctx.amp,
        "class_weighting": tcfg["class_weighting"], "aux_weight": cfg["loss"]["aux_weight"],
        "selection_rule": "highest val macro-F1; ties -> lower val cross-entropy",
        "augment": cfg["data"]["augment"],
    }
    hw = parse_hw(cfg)

    def make_ckpt(epoch: int, extra: dict) -> dict:
        return build_checkpoint(model=model, model_cfg=mcfg, normalization=ctx.norm, class_weights=ctx.class_weights,
                                split_id=cfg["split"]["id"], provenance="trained", name=cfg["name"], input_size=hw,
                                training=training_meta, extra={"epoch": epoch, **extra})

    log_path = out / "log.csv"
    t_start = time.time()
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, tcfg["max_epochs"] + 1):
        seed_everything(cfg["seed"] + epoch)
        ctx.train_loader.generator.manual_seed(cfg["seed"] + epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr = train_epoch(model, ctx.train_loader, crit, opt, scaler, device, ctx.amp, tcfg.get("grad_clip_norm"),
                         log_every=100)
        ev = run_eval(model, ctx.val_loader, device, ctx.amp, cfg["eval"]["mask_threshold"], want_mask_stats=False)
        sched.step()
        cm = confusion_matrix(ev["true_ids"], ev["pred_ids"])
        rep = classification_report(cm)
        insp = inspection_report(ev["true_ids"], ev["pred_ids"])
        f1, ce = rep["macro_f1"], ev["ce_loss"]
        improved = _is_better(f1, ce, best)
        if improved:
            best, bad_epochs = {"f1": f1, "ce": ce, "epoch": epoch}, 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tr["loss"], "train_cls_loss": tr["cls_loss"],
            "train_seg_loss": tr["seg_loss"], "val_ce": ce, "val_macro_f1": f1, "val_accuracy": rep["accuracy"]["rate"],
            "val_false_pass_rate": insp["false_pass"]["rate"], "val_false_fail_rate": insp["false_fail"]["rate"],
            "epoch_seconds": tr["seconds"] + ev["seconds"],
            "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
            "best": improved,
        }
        history.append(row)
        new_file = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} | val CE {ce:.4f} macroF1 {f1:.4f} "
              f"acc {rep['accuracy']['rate']:.4f} | {row['epoch_seconds']:.0f}s{' | *best*' if improved else ''}",
              flush=True)
        if improved:
            save_checkpoint(wdir / "best.pt", make_ckpt(epoch, {"val_macro_f1": f1, "val_ce": ce}))
        save_checkpoint(wdir / "latest.pt", make_ckpt(epoch, {"val_macro_f1": f1, "val_ce": ce}))
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "bad_epochs": bad_epochs, "best": best,
                    "history_json": json.dumps(history)}, resume_path)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if bad_epochs >= tcfg["early_stopping_patience"]:
            stop_reason = f"early_stopping (no improvement for {bad_epochs} epochs)"
            print(f"Early stopping: {stop_reason}")
            break

    summary = {"name": cfg["name"], "split_id": cfg["split"]["id"], "best_epoch": best["epoch"],
               "best_val_macro_f1": best["f1"], "epochs_run": len(history), "stop_reason": stop_reason,
               "wall_seconds_this_session": time.time() - t_start, "class_weights": ctx.class_weights,
               "training": training_meta, "provenance": "trained"}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Best epoch {best['epoch']} (val macro-F1 {best['f1']:.4f}). Checkpoint: {wdir / 'best.pt'}")


if __name__ == "__main__":
    main()
