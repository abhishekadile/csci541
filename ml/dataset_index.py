"""Reading the Carinthia-S index and decoding images / masks.

Carinthia-S layout (verified against the extracted data):
    <DATA_ROOT>/carinthia-s.csv      ';'-separated: image_path;mask_path;filename;label
    <DATA_ROOT>/images/<id>.jpg      480x480, 8-bit grayscale
    <DATA_ROOT>/masks/<id>.png       binary masks with anti-aliased edges (0/255 plus
                                     a few intermediate values); 4 of them are stored as RGB/RGBA.
Masks are binarised with value > 127.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import SOURCE_CLASS_IDS

MASK_BINARY_THRESHOLD = 127
INDEX_CSV_NAMES = ("carinthia-s.csv",)


@dataclass(frozen=True)
class Record:
    filename: str  # id without extension
    label: int  # original source class ID 1..6
    image_path: Path
    mask_path: Path | None


def _resolve(root: Path, rel: str, subdir: str) -> Path:
    p = root / rel
    if p.is_file():
        return p
    alt = root / subdir / Path(rel).name
    return alt


def read_records(data_root: Path, require_masks: bool = True) -> list[Record]:
    index = next((data_root / n for n in INDEX_CSV_NAMES if (data_root / n).is_file()), None)
    if index is None:
        raise SystemExit(
            f"No Carinthia-S index found in {data_root}. Expected one of {INDEX_CSV_NAMES}. "
            "The original Carinthia download (carinthia.csv, no masks) is not sufficient; "
            "get Carinthia-S from https://zenodo.org/records/16895427."
        )
    records: list[Record] = []
    with index.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            label = int(row["label"])
            if label not in SOURCE_CLASS_IDS:
                raise SystemExit(f"Unexpected label {label!r} for {row['filename']}")
            img = _resolve(data_root, row["image_path"], "images")
            mask = _resolve(data_root, row["mask_path"], "masks") if row.get("mask_path") else None
            if require_masks and (mask is None or not mask.is_file()):
                raise SystemExit(f"Mask missing for {row['filename']}: {mask}")
            records.append(Record(row["filename"], label, img, mask))
    if not records:
        raise SystemExit(f"Index {index} has no rows")
    return records


def load_image_u8(path: Path) -> np.ndarray:
    """Grayscale uint8 HxW at native resolution (no resizing, no cropping)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_mask_binary(path: Path) -> np.ndarray:
    """Binary uint8 HxW (values 0/1) at native resolution."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)
    return (arr > MASK_BINARY_THRESHOLD).astype(np.uint8)
