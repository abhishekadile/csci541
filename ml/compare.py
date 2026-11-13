"""Side-by-side comparison of Model A and Model B from saved evaluation results.

    python -m ml.compare

Reads artifacts/runs/<model>/eval_{val,test}.json and predictions_*.csv, writes to
artifacts/results/: comparison.json (served by the API), comparison.md, recall-by-size and
confusion-matrix plots, and a gallery of actual errors (false passes, false fails, wrong
defect types) with expert masks and, for Model B, predicted masks. Nothing is invented: a
missing model or split is reported as missing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .checkpoint import load_checkpoint  # noqa: E402
from .config import artifacts_dir, get_data_root, load_config, run_dir, weights_dir  # noqa: E402
from .dataset_index import load_image_u8, load_mask_binary, read_records  # noqa: E402
from .decision import FALSE_FAIL, FALSE_PASS, WRONG_DEFECT_TYPE  # noqa: E402
from .labels import SOURCE_CLASS_IDS  # noqa: E402
from .predict import Predictor  # noqa: E402

MODELS = (("model_a", "Model A (baseline, class labels only)"), ("model_b", "Model B (mask-supervised)"))
SIZES = ("small", "medium", "large")
KINDS = (FALSE_PASS, FALSE_FAIL, WRONG_DEFECT_TYPE)


def _load(name: str, split: str) -> dict | None:
    p = artifacts_dir() / "runs" / name / f"eval_{split}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _frac(d: dict) -> str:
    return f"{d['num']}/{d['den']} ({_pct(d['rate'])})"


def _table(results: dict, split: str) -> list[str]:
    rows = []
    names = [m for m, _ in MODELS if results.get(m, {}).get(split)]
    if not names:
        return [f"_No evaluation results for the {split} split yet._"]
    head = "| Metric | " + " | ".join(f"{n} / {split}" for n in names) + " |"
    rows += [head, "|---|" + "---|" * len(names)]

    def add(label, fn):
        rows.append(f"| {label} | " + " | ".join(fn(results[n][split]) for n in names) + " |")

    add("Overall accuracy", lambda r: _frac(r["six_class"]["accuracy"]))
    add("Macro-F1", lambda r: f"{r['six_class']['macro_f1']:.4f}")
    add("Majority-class baseline accuracy / macro-F1",
        lambda r: f"{_pct(r['six_class']['majority_class_baseline']['accuracy']['rate'])} / "
                  f"{r['six_class']['majority_class_baseline']['macro_f1']:.4f}")
    for sid in SOURCE_CLASS_IDS:
        add(f"Class {sid} recall", lambda r, sid=sid: r["six_class"]["per_class"][sid - 1]["recall_text"])
        add(f"Class {sid} F1", lambda r, sid=sid: (lambda f: "n/a" if f is None else f"{f:.3f}")(r["six_class"]["per_class"][sid - 1]["f1"]))
    add("False passes (defect -> class 6)", lambda r: _frac(r["inspection"]["false_pass"]))
    add("False fails (class 6 -> defect)", lambda r: _frac(r["inspection"]["false_fail"]))
    add("Wrong defect type (still FAIL)", lambda r: _frac(r["inspection"]["wrong_defect_type"]))
    for g in SIZES:
        add(f"{g} defects flagged FAIL", lambda r, g=g: _frac(r["size_groups"]["groups"][g]["flagged_fail"]))
        add(f"{g} defects exact class", lambda r, g=g: _frac(r["size_groups"]["groups"][g]["exact_class_correct"]))
    add("Mask Dice / IoU (non-empty expert masks)",
        lambda r: "n/a (no decoder)" if not r.get("mask") else f"{r['mask']['mean_dice']:.4f} / {r['mask']['mean_iou']:.4f}")
    return rows


def _plot_sizes(results: dict, split: str, out: Path) -> bool:
    names = [m for m, _ in MODELS if results.get(m, {}).get(split)]
    if not names:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / len(names)
    for ax, key, title in ((axes[0], "flagged_fail", "Flagged FAIL"), (axes[1], "exact_class_correct", "Exact class correct")):
        for k, n in enumerate(names):
            g = results[n][split]["size_groups"]["groups"]
            vals = [100 * (g[s][key]["rate"] or 0) for s in SIZES]
            xs = np.arange(len(SIZES)) + k * width
            bars = ax.bar(xs, vals, width, label=n)
            for b, s in zip(bars, SIZES):
                d = g[s][key]
                ax.annotate(f"{d['num']}/{d['den']}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(SIZES)) + width * (len(names) - 1) / 2, SIZES)
        ax.set_ylim(0, 112)
        ax.set_ylabel("% of defect images (classes 1-5)")
        ax.set_xlabel("Defect size group (train-positive tertiles)")
        ax.set_title(f"{title} by defect size ({split} split)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def _plot_confusion(results: dict, split: str, out: Path) -> bool:
    names = [m for m, _ in MODELS if results.get(m, {}).get(split)]
    if not names:
        return False
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4.8), squeeze=False)
    for ax, n in zip(axes[0], names):
        cm = np.array(results[n][split]["six_class"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(6), SOURCE_CLASS_IDS)
        ax.set_yticks(range(6), SOURCE_CLASS_IDS)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{n} | {split}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _gallery(cfg: dict, results: dict, out_dir: Path, split: str = "test") -> list[dict]:
    by_name = {r.filename: r for r in read_records(get_data_root(), require_masks=True)}
    entries: list[dict] = []
    k = int(cfg["eval"]["gallery_per_kind"])
    for model, _ in MODELS:
        if not results.get(model, {}).get(split):
            continue
        rd = run_dir(load_config(f"{model}.yaml"))
        with (rd / f"predictions_{split}.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        predictor = None
        has_decoder = results[model][split].get("mask") is not None
        if has_decoder:
            predictor = Predictor.from_path(weights_dir(load_config(f"{model}.yaml")) / "best.pt", "cpu")
        for kind in KINDS:
            errs = sorted((r for r in rows if r["outcome"] == kind), key=lambda r: -float(r["predicted_class_score"]))[:k]
            for r in errs:
                rec = by_name[r["filename"]]
                base = f"gallery/{model}/{kind}/{r['filename']}"
                _save_png(load_image_u8(rec.image_path), out_dir / f"{base}_image.png")
                _save_png(load_mask_binary(rec.mask_path) * 255, out_dir / f"{base}_expert.png")
                pred_asset = None
                if predictor is not None:
                    pm = predictor.predict_u8(load_image_u8(rec.image_path))["predicted_mask"]
                    _save_png(pm * 255, out_dir / f"{base}_predicted.png")
                    pred_asset = f"{base}_predicted.png"
                entries.append({
                    "model": model, "split": split, "kind": kind, "filename": r["filename"],
                    "true_class_id": int(r["true_class_id"]), "predicted_class_id": int(r["predicted_class_id"]),
                    "inspection_decision": r["inspection_decision"], "reference_decision": r["reference_decision"],
                    "predicted_class_score": float(r["predicted_class_score"]),
                    "class_scores": {str(s): float(r[f"score_class_{s}"]) for s in SOURCE_CLASS_IDS},
                    "image": f"{base}_image.png", "expert_mask": f"{base}_expert.png", "predicted_mask": pred_asset,
                })
    return entries


def main() -> None:
    cfg = load_config()
    out_dir = artifacts_dir() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {m: {s: _load(m, s) for s in ("val", "test")} for m, _ in MODELS}
    available = {m: [s for s, r in v.items() if r] for m, v in results.items()}
    plots = {}
    for split in ("val", "test"):
        if _plot_sizes(results, split, out_dir / f"recall_by_size_{split}.png"):
            plots[f"recall_by_size_{split}"] = f"recall_by_size_{split}.png"
        if _plot_confusion(results, split, out_dir / f"confusion_{split}.png"):
            plots[f"confusion_{split}"] = f"confusion_{split}.png"
    gallery = _gallery(cfg, results, out_dir) if any("test" in v for v in available.values()) else []
    split_report = artifacts_dir() / "splits" / cfg["split"]["id"] / "split_report.json"
    comparison = {
        "split_id": cfg["split"]["id"],
        "models": {m: {"label": lbl, "available_splits": available[m],
                       "model_version": (results[m]["test"] or results[m]["val"] or {}).get("model_version")}
                   for m, lbl in MODELS},
        "results": results, "plots": plots, "gallery": gallery,
        "split_report": json.loads(split_report.read_text()) if split_report.is_file() else None,
        "notes": "Every metric belongs to the model and split named in its container. Test results are from the frozen best checkpoint.",
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    md = ["# Model comparison", "", f"Split: `{cfg['split']['id']}`", ""]
    for split in ("val", "test"):
        md += [f"## {split} split", "", *_table(results, split), ""]
    (out_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_dir / 'comparison.json'}, comparison.md, {len(plots)} plots, {len(gallery)} gallery entries")
    print(f"Available results: {available}")


if __name__ == "__main__":
    main()
