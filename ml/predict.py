"""Inference core shared by evaluation, the CLI and the API.

    python -m ml.predict --checkpoint weight/model_b/best.pt --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import numpy as np
import torch

from .checkpoint import LoadedCheckpoint, load_checkpoint
from .data import normalize_u8
from .dataset_index import load_image_u8
from .decision import decide
from .labels import SOURCE_CLASS_IDS, class_name


def decode_outputs(logits: torch.Tensor, mask_logits: torch.Tensor | None, index_to_source: list[int],
                   mask_threshold: float = 0.5) -> list[dict]:
    """Model outputs (internal indices) -> per-image results in ORIGINAL source class IDs."""
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    results = []
    for b in range(probs.shape[0]):
        idx = int(probs[b].argmax())
        pid = int(index_to_source[idx])
        scores = {str(index_to_source[i]): float(probs[b, i]) for i in range(probs.shape[1])}
        scores = {str(sid): scores[str(sid)] for sid in SOURCE_CLASS_IDS}
        d = decide(pid)
        res = {
            "predicted_class_id": pid,
            "predicted_class_name": class_name(pid),
            "predicted_class_score": float(probs[b, idx]),
            "class_scores": scores,
            **d.as_dict(),
            "predicted_mask": None,
        }
        if mask_logits is not None:
            res["predicted_mask"] = (torch.sigmoid(mask_logits[b, 0].float()) > mask_threshold).cpu().numpy().astype(np.uint8)
        results.append(res)
    return results


class Predictor:
    """Holds one loaded checkpoint; thread-safe."""

    def __init__(self, ckpt: LoadedCheckpoint, device: str = "cpu", mask_threshold: float = 0.5):
        self.ckpt = ckpt
        self.device = torch.device(device)
        self.model = ckpt.model.to(self.device).eval()
        self.mask_threshold = mask_threshold
        self.mean = float(ckpt.meta["normalization"]["mean"])
        self.std = float(ckpt.meta["normalization"]["std"])
        self.has_mask_output = bool(ckpt.meta["model"]["with_decoder"])
        self.input_size = tuple(ckpt.meta["input_size"])
        self._lock = threading.Lock()

    @classmethod
    def from_path(cls, path: str | Path, device: str = "cpu", allow_non_trained: bool = False) -> "Predictor":
        return cls(load_checkpoint(path, device, allow_non_trained), device)

    @property
    def model_version(self) -> str:
        return str(self.ckpt.meta["model_version"])

    @torch.no_grad()
    def predict_u8(self, img_u8: np.ndarray) -> dict:
        """img_u8: HxW uint8 grayscale at native resolution (no resize is applied)."""
        x = normalize_u8(img_u8, self.mean, self.std).unsqueeze(0).to(self.device)
        with self._lock:
            out = self.model(x)
        res = decode_outputs(out["logits"], out["mask_logits"], self.ckpt.index_to_source, self.mask_threshold)[0]
        res["model_version"] = self.model_version
        return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-non-trained", action="store_true", help="smoke/test checkpoints only")
    ap.add_argument("--save-mask", help="write predicted mask PNG (Model B only)")
    args = ap.parse_args()
    p = Predictor.from_path(args.checkpoint, args.device, args.allow_non_trained)
    res = p.predict_u8(load_image_u8(Path(args.image)))
    mask = res.pop("predicted_mask")
    if mask is not None and args.save_mask:
        from PIL import Image
        Image.fromarray(mask * 255).save(args.save_mask)
    res["predicted_mask_pixels"] = int(mask.sum()) if mask is not None else None
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
