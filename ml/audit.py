"""Dataset audit: paths, dimensions, label counts, masks, duplicates.

    python -m ml.audit [--refresh]

Writes artifacts/audit/audit.json and a feature cache reused by ml.split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from .config import artifacts_dir, get_data_root, load_config
from .dataset_index import MASK_BINARY_THRESHOLD, Record, load_mask_binary, read_records
from .labels import PASS_CLASS_ID, SOURCE_CLASS_IDS


def _dhash(img: Image.Image) -> int:
    small = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def compute_features(records: list[Record], refresh: bool = False) -> dict:
    """Per-image features (cached): md5, dhash, 32x32 thumbnail, pixel sums, mask stats."""
    cache = artifacts_dir() / "audit" / "features.npz"
    names = np.array([r.filename for r in records])
    if cache.is_file() and not refresh:
        z = np.load(cache, allow_pickle=False)
        if np.array_equal(z["names"], names):
            return {k: z[k] for k in z.files}
    n = len(records)
    feats = {
        "names": names,
        "md5": np.empty(n, dtype="U32"),
        "dhash": np.zeros(n, dtype=np.uint64),
        "thumb": np.zeros((n, 32 * 32), dtype=np.float32),
        "height": np.zeros(n, dtype=np.int32),
        "width": np.zeros(n, dtype=np.int32),
        "pix_sum": np.zeros(n, dtype=np.float64),
        "pix_sumsq": np.zeros(n, dtype=np.float64),
        "mask_area": np.zeros(n, dtype=np.float64),  # fraction of native pixels
        "mask_pixels": np.zeros(n, dtype=np.int64),
        "mask_mode_rgb": np.zeros(n, dtype=bool),
        "mask_size_ok": np.zeros(n, dtype=bool),
    }
    t0 = time.time()
    for i, r in enumerate(records):
        feats["md5"][i] = hashlib.md5(r.image_path.read_bytes()).hexdigest()
        with Image.open(r.image_path) as im:
            im.load()
            g = im.convert("L")
            feats["height"][i], feats["width"][i] = g.size[1], g.size[0]
            arr = np.asarray(g, dtype=np.float64) / 255.0
            feats["pix_sum"][i] = arr.sum()
            feats["pix_sumsq"][i] = (arr**2).sum()
            feats["dhash"][i] = np.uint64(_dhash(g))
            feats["thumb"][i] = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.float32).flatten()
        if r.mask_path is not None:
            with Image.open(r.mask_path) as m:
                feats["mask_mode_rgb"][i] = m.mode != "L"
                feats["mask_size_ok"][i] = m.size == (int(feats["width"][i]), int(feats["height"][i]))
            mask = load_mask_binary(r.mask_path)
            feats["mask_pixels"][i] = int(mask.sum())
            feats["mask_area"][i] = float(mask.mean())
        if (i + 1) % 500 == 0:
            print(f"  features {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **feats)
    return feats


def find_duplicate_groups(feats: dict, dhash_max: int, min_corr: float) -> tuple[np.ndarray, dict]:
    """Union-find over exact (md5) and near-duplicate (dHash + thumbnail correlation) pairs.

    Returns (group_id per image, stats)."""
    n = len(feats["names"])
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(feats["md5"]):
        by_md5[str(h)].append(i)
    exact_pairs = 0
    for idxs in by_md5.values():
        for j in idxs[1:]:
            union(idxs[0], j)
            exact_pairs += 1

    thumbs = feats["thumb"].astype(np.float64)
    thumbs = thumbs - thumbs.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(thumbs, axis=1)
    norms[norms == 0] = 1.0
    thumbs = thumbs / norms[:, None]
    dh = feats["dhash"]
    near_pairs = 0
    for i in range(n - 1):
        dist = np.bitwise_count(dh[i] ^ dh[i + 1 :])
        cand = np.nonzero(dist <= dhash_max)[0] + i + 1
        for j in cand:
            if float(thumbs[i] @ thumbs[j]) >= min_corr:
                if find(i) != find(int(j)):
                    near_pairs += 1
                union(i, int(j))
    roots = np.array([find(i) for i in range(n)])
    uniq = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
    gid = np.array([uniq[r] for r in roots], dtype=np.int64)
    return gid, {"exact_duplicate_extra_copies": exact_pairs, "near_duplicate_merges": near_pairs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="recompute cached features")
    args = ap.parse_args()
    cfg = load_config()
    root = get_data_root()
    records = read_records(root, require_masks=True)
    labels = np.array([r.label for r in records])
    print(f"DATA_ROOT: {root}")
    print(f"Index rows: {len(records)}   images on disk: {sum(r.image_path.is_file() for r in records)}   "
          f"masks on disk: {sum(r.mask_path is not None and r.mask_path.is_file() for r in records)}")
    feats = compute_features(records, refresh=args.refresh)

    sizes = Counter(zip(feats["height"].tolist(), feats["width"].tolist()))
    print(f"Image sizes (HxW): {dict(sizes)}   expected input_size={cfg['data']['input_size']}")
    counts = {int(c): int((labels == c).sum()) for c in SOURCE_CLASS_IDS}
    print(f"Class counts (source IDs): {counts}")
    print(f"Documented meaning: class {PASS_CLASS_ID} = 'no visible defects'; no descriptive names published for 1-5")

    area = feats["mask_area"]
    npix = feats["mask_pixels"]
    print(f"Masks: binarised with value > {MASK_BINARY_THRESHOLD}; "
          f"non-grayscale mask files: {int(feats['mask_mode_rgb'].sum())}; "
          f"mask size != image size: {int((~feats['mask_size_ok']).sum())}")
    per_class = {}
    for c in SOURCE_CLASS_IDS:
        m = labels == c
        empty = int((npix[m] == 0).sum())
        per_class[c] = {
            "n": int(m.sum()),
            "empty_masks": empty,
            "area_fraction_min": float(area[m].min()),
            "area_fraction_median": float(np.median(area[m])),
            "area_fraction_max": float(area[m].max()),
        }
        print(f"  class {c}: n={int(m.sum()):5d} empty masks={empty:4d} area min/median/max = "
              f"{area[m].min():.5f} / {np.median(area[m]):.5f} / {area[m].max():.5f}")
    defect_empty = [records[i].filename for i in range(len(records))
                    if labels[i] != PASS_CLASS_ID and npix[i] == 0]
    pass_nonempty = [(records[i].filename, int(npix[i])) for i in range(len(records))
                     if labels[i] == PASS_CLASS_ID and npix[i] > 0]
    print(f"Label/mask disagreements: defect classes 1-5 with EMPTY mask = {len(defect_empty)}; "
          f"class 6 with NON-EMPTY mask = {len(pass_nonempty)} {pass_nonempty}")

    gid, dstats = find_duplicate_groups(
        feats, cfg["split"]["near_duplicate"]["dhash_max_distance"], cfg["split"]["near_duplicate"]["min_correlation"]
    )
    group_sizes = Counter(gid.tolist())
    multi = [g for g, s in group_sizes.items() if s > 1]
    conflict = [g for g in multi if len(set(labels[gid == g].tolist())) > 1]
    md5_dups = len(feats["md5"]) - len(set(feats["md5"].tolist()))
    print(f"Duplicates: exact (md5) extra copies = {md5_dups}; near-duplicate groups = {len(multi)} "
          f"covering {sum(group_sizes[g] for g in multi)} images; groups with conflicting labels = {len(conflict)}")
    print("Identifiers: no wafer / acquisition-group IDs in the index -> evaluation is IMAGE-LEVEL "
          "(near-duplicate groups are kept together).")
    px = (feats["pix_sum"].sum() / (feats["height"] * feats["width"]).sum())
    print(f"Mean pixel intensity (all images, [0,1]) = {px:.4f}")

    out = artifacts_dir() / "audit" / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "data_root": str(root), "n_records": len(records), "image_sizes": {f"{h}x{w}": c for (h, w), c in sizes.items()},
        "class_counts": counts, "per_class_masks": per_class,
        "defect_with_empty_mask": defect_empty, "pass_with_nonempty_mask": pass_nonempty,
        "duplicates": {"exact_extra_copies": md5_dups, "near_duplicate_groups": len(multi),
                       "images_in_near_duplicate_groups": int(sum(group_sizes[g] for g in multi)),
                       "label_conflicting_groups": len(conflict), **dstats},
        "group_identifiers_available": False,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
