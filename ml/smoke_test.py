"""Smoke test: real image/mask pairs, forward + backward pass for BOTH models.

    python -m ml.smoke_test

Everything written goes to artifacts/smoke/ and is marked provenance="smoke"; the checkpoint
loader and the API refuse such files as production models. This is not training.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from .checkpoint import CheckpointError, build_checkpoint, load_checkpoint, save_checkpoint
from .config import artifacts_dir, get_data_root, load_config
from .data import WaferDataset, pad_collate
from .dataset_index import load_image_u8, read_records
from .engine import autocast_ctx, pick_device, to_device
from .labels import NUM_CLASSES, SOURCE_CLASS_IDS
from .losses import MultiTaskLoss, compute_class_weights
from .models import build_model
from .predict import Predictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--per-class", type=int, default=2, help="real images per class (default 2 -> 12 images)")
    args = ap.parse_args()

    device = pick_device(args.device)
    records = read_records(get_data_root(), require_masks=True)
    chosen = []
    for c in SOURCE_CLASS_IDS:
        chosen += [r for r in records if r.label == c][: args.per_class]
    print(f"Smoke test on {len(chosen)} REAL image/mask pairs (classes {[r.label for r in chosen]}), device={device}")
    mean, std = 0.3777, 0.0497
    ds = WaferDataset(chosen, mean, std, with_masks=True,
                      augment=load_config()["data"]["augment"])  # exercises augmentation too
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=pad_collate)
    report = {"smoke": True, "provenance": "smoke", "device": str(device), "models": {},
              "warning": "Smoke-test artifact. Not a trained model; never load for production."}
    out_dir = artifacts_dir() / "smoke"
    ok = True
    for name in ("model_a.yaml", "model_b.yaml"):
        cfg = load_config(name, "smoke.yaml")
        model = build_model(cfg["model"]).to(device)
        crit = MultiTaskLoss(compute_class_weights([0, 1, 2, 3, 4, 5] * 3, "sqrt_inverse"),
                             cfg["loss"]["aux_weight"], device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["train"]["amp"])
        batch = to_device(next(iter(loader)), device)
        assert batch["image"].shape[-2:] == (480, 480), "native resolution must be preserved"
        t0 = time.time()
        model.train()
        opt.zero_grad()
        with autocast_ctx(device, cfg["train"]["amp"]):
            out = model(batch["image"])
        losses = crit(out, batch["label_index"], batch["mask"])
        scaler.scale(losses["total"]).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_ok = bool(grads) and all(torch.isfinite(g).all() for g in grads)
        scaler.step(opt)
        scaler.update()
        shape_ok = tuple(out["logits"].shape) == (batch["image"].shape[0], NUM_CLASSES)
        mask_ok = (out["mask_logits"] is None) == (not cfg["model"]["with_decoder"])
        if out["mask_logits"] is not None:
            mask_ok = mask_ok and tuple(out["mask_logits"].shape[-2:]) == (480, 480)
        loss_ok = math.isfinite(float(losses["total"]))
        entry = {"loss": float(losses["total"]), "logits_shape": list(out["logits"].shape),
                 "mask_logits_shape": list(out["mask_logits"].shape) if out["mask_logits"] is not None else None,
                 "grads_finite": grad_ok, "shape_ok": shape_ok, "mask_output_ok": mask_ok, "loss_finite": loss_ok,
                 "seconds": time.time() - t0}
        # smoke checkpoint: must be written as non-production and refused by the default loader
        ckpt = build_checkpoint(model=model, model_cfg=cfg["model"], normalization={"mean": mean, "std": std},
                                class_weights=[1.0] * NUM_CLASSES, split_id="smoke", provenance="smoke",
                                name=cfg["name"], input_size=(480, 480))
        path = out_dir / cfg["name"] / "smoke_checkpoint.pt"
        save_checkpoint(path, ckpt)
        try:
            load_checkpoint(path, "cpu")
            entry["refused_as_production_model"] = False
        except CheckpointError:
            entry["refused_as_production_model"] = True
        pred = Predictor.from_path(path, "cpu", allow_non_trained=True).predict_u8(
            load_image_u8(ds.records[0].image_path))
        entry["decode_roundtrip_ok"] = pred["predicted_class_id"] in SOURCE_CLASS_IDS and \
            abs(sum(pred["class_scores"].values()) - 1.0) < 1e-4 and \
            (pred["predicted_mask"] is not None) == cfg["model"]["with_decoder"]
        entry["passed"] = all([grad_ok, shape_ok, mask_ok, loss_ok, entry["refused_as_production_model"],
                               entry["decode_roundtrip_ok"]])
        ok &= entry["passed"]
        report["models"][cfg["name"]] = entry
        print(f"  {cfg['name']}: loss={entry['loss']:.4f} logits={entry['logits_shape']} "
              f"mask={entry['mask_logits_shape']} grads_finite={grad_ok} smoke-ckpt refused={entry['refused_as_production_model']} "
              f"=> {'PASS' if entry['passed'] else 'FAIL'}")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(f"Smoke test {'PASSED' if ok else 'FAILED'} (report: {out_dir / 'smoke_report.json'})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
