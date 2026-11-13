"""Evaluate one trained model on one split.

    python -m ml.evaluate model_a.yaml --split val
    python -m ml.evaluate model_a.yaml --split test      # once per frozen checkpoint

Six-class results and the derived PASS/FAIL inspection results are reported separately.
Predictions are decoded back to ORIGINAL source class IDs before any metric is computed,
and the PASS/FAIL decision comes from ml.decision (the same function the API uses).
The test split requires a finished training run (run_summary.json) and refuses to be
re-run on the same checkpoint unless --force is given.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np

from .checkpoint import load_checkpoint
from .config import get_data_root, load_config, run_dir, split_dir, weights_dir
from .data import WaferDataset, make_loader, records_for_split
from .dataset_index import read_records
from .decision import categorize, decide, reference_decision
from .engine import pick_device, run_eval
from .labels import SOURCE_CLASS_IDS
from .metrics import (classification_report, confusion_matrix, inspection_report, majority_baseline, mask_report,
                      size_group_report)
from .split import load_manifest, size_group


def evaluate(cfg: dict, split: str, device_name: str = "auto", force: bool = False,
             allow_unfinished: bool = False, batch_size: int | None = None) -> dict:
    out = run_dir(cfg)
    ckpt_path = weights_dir(cfg) / "best.pt"
    result_path = out / f"eval_{split}.json"
    if split == "test":
        if not (out / "run_summary.json").is_file() and not allow_unfinished:
            raise SystemExit(f"Training for {cfg['name']} has not finished (no run_summary.json). "
                             "Freeze the checkpoint first (finish training), then evaluate on test.")
    device = pick_device(device_name)
    loaded = load_checkpoint(ckpt_path, device)  # refuses smoke / fixture / binary checkpoints
    if split == "test" and result_path.is_file() and not force:
        prev = json.loads(result_path.read_text())
        if prev.get("checkpoint_sha256") == loaded.sha256:
            raise SystemExit(f"{result_path} already exists for this frozen checkpoint. "
                             "The test split is evaluated once; use --force to overwrite.")
    manifest = load_manifest(cfg)
    by_name = {r.filename: r for r in read_records(get_data_root(), require_masks=True)}
    sdir = split_dir(cfg)
    cutoffs = json.loads((sdir / "size_cutoffs.json").read_text())
    recs = records_for_split(manifest, by_name, split)
    area = {r["filename"]: r["area_fraction"] for r in manifest}
    norm = loaded.meta["normalization"]
    ds = WaferDataset(recs, norm["mean"], norm["std"], with_masks=True, augment=None)
    bs = batch_size or cfg["train"]["batch_size"]
    loader = make_loader(ds, bs, False, cfg["data"]["num_workers"], cfg["seed"], device.type == "cuda")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    ev = run_eval(loaded.model, loader, device, amp, cfg["eval"]["mask_threshold"])

    y_true, y_pred = ev["true_ids"], ev["pred_ids"]
    cm = confusion_matrix(y_true, y_pred)
    six = classification_report(cm)
    train_labels = [r["label"] for r in manifest if r["split"] == "train"]
    maj = majority_baseline(train_labels, y_true)
    insp = inspection_report(y_true, y_pred)

    pos = [i for i, t in enumerate(y_true) if t != SOURCE_CLASS_IDS[-1]]
    groups = [size_group(area[ev["filenames"][i]], cutoffs) for i in pos]
    sizes = size_group_report([area[ev["filenames"][i]] for i in pos], [y_true[i] for i in pos],
                              [y_pred[i] for i in pos], groups)

    masks = mask_report(ev["mask_items"], y_true) if ev["mask_items"] else None
    pred_rows = []
    for i, fn in enumerate(ev["filenames"]):
        t, p = y_true[i], y_pred[i]
        mi = ev["mask_items"][i] if ev["mask_items"] else None
        pred_rows.append({
            "filename": fn, "true_class_id": t, "predicted_class_id": p,
            "reference_decision": reference_decision(t), "inspection_decision": decide(p).decision,
            "outcome": categorize(t, p), "predicted_class_score": float(ev["probs"][i].max()),
            **{f"score_class_{sid}": float(ev["probs"][i, k]) for k, sid in enumerate(SOURCE_CLASS_IDS)},
            "defect_area_fraction": area[fn],
            "size_group": size_group(area[fn], cutoffs) if t != SOURCE_CLASS_IDS[-1] else "",
            "mask_dice": (2 * mi["inter"] / (mi["pred"] + mi["gt"])) if mi and (mi["pred"] + mi["gt"]) else "",
            "mask_iou": (mi["inter"] / mi["union"]) if mi and mi["union"] else "",
            "pred_mask_pixels": mi["pred"] if mi else "", "expert_mask_pixels": mi["gt"] if mi else "",
        })

    result = {
        "model": cfg["name"], "model_kind": cfg["model"]["kind"], "model_version": loaded.meta["model_version"],
        "split": split, "split_id": cfg["split"]["id"], "n_images": ev["n"],
        "checkpoint_sha256": loaded.sha256, "checkpoint_epoch": loaded.meta.get("epoch"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_rule": loaded.meta["decision_rule"],
        "class_weights": loaded.meta["class_weights"], "aux_weight": cfg["loss"]["aux_weight"],
        "six_class": {**six, "confusion_matrix": cm.tolist(), "confusion_matrix_order": list(SOURCE_CLASS_IDS),
                      "confusion_matrix_axes": "rows = true class, columns = predicted class (original source IDs)",
                      "majority_class_baseline": maj},
        "inspection": insp,
        "size_groups": {"cutoffs": cutoffs, "groups": sizes},
        "mask": masks,
    }
    out.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
        w.writeheader()
        w.writerows(pred_rows)
    return result


def print_result(r: dict) -> None:
    tag = f"[{r['model']} | {r['model_version']} | {r['split']} (n={r['n_images']})]"
    s = r["six_class"]
    print(f"{tag} accuracy {s['accuracy']['num']}/{s['accuracy']['den']} = {s['accuracy']['rate']:.4f}; "
          f"macro-F1 {s['macro_f1']:.4f} over classes {s['macro_f1_classes']}; "
          f"majority baseline acc {s['majority_class_baseline']['accuracy']['rate']:.4f} "
          f"macro-F1 {s['majority_class_baseline']['macro_f1']:.4f}")
    for e in s["per_class"]:
        f1 = f"{e['f1']:.3f}" if e["f1"] is not None else "n/a"
        print(f"   class {e['class_id']}: {e['recall_text']}; F1 {f1}")
    i = r["inspection"]
    for k in ("false_pass", "false_fail", "wrong_defect_type"):
        print(f"   {k}: {i[k]['num']}/{i[k]['den']}"
              + (f" = {i[k]['rate']:.4f}" if i[k]["rate"] is not None else ""))
    for g, v in r["size_groups"]["groups"].items():
        print(f"   size {g}: n={v['n']} flagged FAIL {v['flagged_fail']['num']}/{v['flagged_fail']['den']}; "
              f"exact class {v['exact_class_correct']['num']}/{v['exact_class_correct']['den']}")
    if r["mask"]:
        print(f"   mask Dice {r['mask']['mean_dice']:.4f} IoU {r['mask']['mean_iou']:.4f} "
              f"(n={r['mask']['images_with_expert_mask']}); empty-mask images with predicted pixels: "
              f"{r['mask']['empty_mask_images_with_any_predicted_pixels']['num']}/"
              f"{r['mask']['empty_mask_images_with_any_predicted_pixels']['den']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-unfinished", action="store_true", help="evaluate test before training finished (not for reporting)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    print_result(evaluate(cfg, a.split, a.device, a.force, a.allow_unfinished, a.batch_size))


if __name__ == "__main__":
    main()
