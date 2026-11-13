"""TEST FIXTURES ONLY.

Builds tiny random-weight checkpoints marked provenance="test_fixture". They exercise the
load/predict/API code paths and carry no information about real model quality. Production
loaders refuse them unless explicitly told otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.checkpoint import build_checkpoint, save_checkpoint
from ml.labels import NUM_CLASSES
from ml.models import build_model


def make_checkpoint(path: Path, with_decoder: bool, provenance: str = "test_fixture", seed: int = 0) -> Path:
    torch.manual_seed(seed)
    cfg = {"kind": "mask_supervised" if with_decoder else "baseline", "with_decoder": with_decoder}
    model = build_model(cfg)
    ckpt = build_checkpoint(model=model, model_cfg=cfg, normalization={"mean": 0.3777, "std": 0.0497},
                            class_weights=[1.0] * NUM_CLASSES, split_id="fixture", provenance=provenance,
                            name="model_b" if with_decoder else "model_a", input_size=(480, 480))
    save_checkpoint(path, ckpt)
    return path


def make_png_bytes(size=(480, 480), fmt="PNG", seed: int = 0) -> bytes:
    import io
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 140, size=(size[1], size[0]), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format=fmt)
    return buf.getvalue()
