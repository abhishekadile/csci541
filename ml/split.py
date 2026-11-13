"""Reproducible 70/15/15 split manifest.

    python -m ml.split [--force]

* Exact (MD5) and near duplicates are grouped; a group never straddles splits.
* No wafer/acquisition IDs exist in the dataset, so the split is image-level.
* Stratified per class as far as the real counts allow (rare classes get at least one
  validation and one test group when they have >= 3 groups).
* Train-only statistics are saved next to the manifest: normalization mean/std and the
  defect-size cutoffs (tertiles of positive TRAIN images' mask area fraction).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

import numpy as np

from .audit import compute_features, find_duplicate_groups
from .config import get_data_root, load_config, split_dir
from .dataset_index import read_records
from .labels import PASS_CLASS_ID, SOURCE_CLASS_IDS

SPLITS = ("train", "val", "test")
SIZE_GROUPS = ("small", "medium", "large")


def size_group(area: float, cutoffs: dict) -> str:
    if area <= cutoffs["small_max"]:
        return "small"
    if area <= cutoffs["medium_max"]:
        return "medium"
    return "large"


def _targets(n_images: int, fr_val: float, fr_test: float) -> tuple[int, int]:
    """Images wanted in val/test for one class (only when the class is big enough)."""
    if n_images >= 3:
        return max(1, round(fr_val * n_images)), max(1, round(fr_test * n_images))
    if n_images == 2:
        return 0, 1
    return 0, 0


def make_split(cfg: dict, records, feats) -> dict:
    sp = cfg["split"]
    rng = np.random.default_rng(sp["seed"])
    labels = np.array([r.label for r in records])
    gid, dstats = find_duplicate_groups(feats, sp["near_duplicate"]["dhash_max_distance"],
                                        sp["near_duplicate"]["min_correlation"])
    assign = np.empty(len(records), dtype=object)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(gid):
        members[int(g)].append(i)
    group_label = {g: int(np.bincount(labels[idx]).argmax()) for g, idx in members.items()}
    conflicts = sum(len(set(labels[idx].tolist())) > 1 for idx in members.values())

    for c in SOURCE_CLASS_IDS:
        groups = [g for g, lab in group_label.items() if lab == c]
        groups.sort()
        rng.shuffle(groups)
        n_img = sum(len(members[g]) for g in groups)
        want_val, want_test = _targets(n_img, sp["fractions"]["val"], sp["fractions"]["test"])
        got = {"val": 0, "test": 0}
        for g in groups:
            size = len(members[g])
            # A duplicate group is never split. A group larger than a split's quota would blow
            # that split up (class 6 has one group of ~157 near-identical images), so it goes to train.
            if got["val"] < want_val and size <= want_val:
                dest = "val"
            elif got["test"] < want_test and size <= want_test:
                dest = "test"
            else:
                dest = "train"
            if dest != "train":
                got[dest] += len(members[g])
            for i in members[g]:
                assign[i] = dest

    area = feats["mask_area"]
    train_pos = [i for i in range(len(records)) if assign[i] == "train" and labels[i] != PASS_CLASS_ID]
    q1, q2 = np.percentile(area[train_pos], [100 / 3, 200 / 3])
    cutoffs = {
        "small_max": float(q1), "medium_max": float(q2),
        "definition": "area = fraction of native-resolution pixels covered by the expert mask; "
                      "small <= small_max < medium <= medium_max < large; cutoffs are the 33.3rd/66.7th "
                      "percentiles of POSITIVE (classes 1-5) TRAIN images only and are applied unchanged "
                      "to validation and test.",
        "n_positive_train_images": len(train_pos),
    }
    tr = [i for i in range(len(records)) if assign[i] == "train"]
    mean = float(feats["pix_sum"][tr].sum() / (feats["height"][tr] * feats["width"][tr]).sum())
    var = float(feats["pix_sumsq"][tr].sum() / (feats["height"][tr] * feats["width"][tr]).sum() - mean**2)
    norm = {"mean": mean, "std": float(math.sqrt(var)), "scale": "pixels / 255, then (x - mean) / std; train split only"}

    report = {
        "split_id": sp["id"], "seed": sp["seed"], "fractions_target": sp["fractions"],
        "unit": "image-level (no wafer or acquisition-group identifiers exist in the dataset)",
        "duplicates": {**dstats, "label_conflicting_groups": conflicts,
                       "n_groups": len(members), "n_images": len(records)},
        "counts_per_class": {}, "distinct_originals_per_class": {}, "size_group_counts": {},
    }
    for s in SPLITS:
        sel = np.array([a == s for a in assign])
        report["counts_per_class"][s] = {int(c): int((labels[sel] == c).sum()) for c in SOURCE_CLASS_IDS}
        report["counts_per_class"][s]["total"] = int(sel.sum())
        report["distinct_originals_per_class"][s] = {
            int(c): len({int(gid[i]) for i in np.nonzero(sel & (labels == c))[0]}) for c in SOURCE_CLASS_IDS
        }
        pos = np.nonzero(sel & (labels != PASS_CLASS_ID))[0]
        report["size_group_counts"][s] = {g: sum(size_group(area[i], cutoffs) == g for i in pos) for g in SIZE_GROUPS}
    return {"assign": assign, "gid": gid, "cutoffs": cutoffs, "norm": norm, "report": report}


def print_report(r: dict, cutoffs: dict, norm: dict) -> None:
    print(f"Split {r['split_id']} (seed {r['seed']}); unit: {r['unit']}")
    print(f"Duplicate groups: {r['duplicates']}")
    hdr = "split   " + " ".join(f"c{c:<5d}" for c in SOURCE_CLASS_IDS) + " total   (images / distinct originals)"
    print(hdr)
    def get(d: dict, k: int) -> int:  # keys are ints when freshly built, strings after a JSON round-trip
        return d[k] if k in d else d[str(k)]

    for s in SPLITS:
        cells = " ".join(f"{get(r['counts_per_class'][s], c):>3d}/{get(r['distinct_originals_per_class'][s], c):<3d}"
                         for c in SOURCE_CLASS_IDS)
        print(f"{s:<7s} {cells} {r['counts_per_class'][s]['total']:>5d}")
    print(f"Size cutoffs (area fraction, train positives): small <= {cutoffs['small_max']:.5f} < medium <= "
          f"{cutoffs['medium_max']:.5f} < large")
    for s in SPLITS:
        print(f"  {s}: {r['size_group_counts'][s]}")
    print(f"Train normalization: mean={norm['mean']:.4f} std={norm['std']:.4f}")


def load_manifest(cfg: dict) -> list[dict]:
    path = split_dir(cfg) / "manifest.csv"
    if not path.is_file():
        raise SystemExit(f"Split manifest missing: {path}\nCreate it with: python -m ml.split")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["area_fraction"] = float(r["area_fraction"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()
    cfg = load_config()
    out = split_dir(cfg)
    if (out / "manifest.csv").is_file() and not args.force:
        r = json.loads((out / "split_report.json").read_text())
        print(f"Manifest already exists: {out} (use --force to rebuild; both models must share one manifest)")
        print_report(r, json.loads((out / "size_cutoffs.json").read_text()),
                     json.loads((out / "normalization.json").read_text()))
        return
    records = read_records(get_data_root(), require_masks=True)
    feats = compute_features(records)
    res = make_split(cfg, records, feats)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "split", "group_id", "area_fraction"])
        for i, rec in enumerate(records):
            w.writerow([rec.filename, rec.label, res["assign"][i], int(res["gid"][i]), f"{feats['mask_area'][i]:.8f}"])
    (out / "size_cutoffs.json").write_text(json.dumps(res["cutoffs"], indent=2), encoding="utf-8")
    (out / "normalization.json").write_text(json.dumps(res["norm"], indent=2), encoding="utf-8")
    (out / "split_report.json").write_text(json.dumps(res["report"], indent=2), encoding="utf-8")
    print_report(res["report"], res["cutoffs"], res["norm"])
    print(f"Wrote manifest and statistics to {out}")


if __name__ == "__main__":
    main()
