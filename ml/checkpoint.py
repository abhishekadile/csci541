"""Checkpoint schema v2 (six-class task) and a safe loader.

A checkpoint carries everything needed to reproduce inference in another process:
weights, model config, normalization, label mapping, class weights, split ID, decision rule.
There is no binary head and no decision threshold; older binary checkpoints are rejected.

`provenance` is one of:
    "trained"      real training run (the only kind that production code loads)
    "smoke"        written by the smoke test / benchmark; never a production model
    "test_fixture" tiny random-weight model created by the test-suite
Smoke and fixture checkpoints are refused unless the caller opts in explicitly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .decision import DECISION_RULE
from .labels import LabelError, NUM_CLASSES, mapping_for_checkpoint, validate_checkpoint_mapping
from .models import SixClassNet, build_model

TASK = "six_class_v1"
SCHEMA_VERSION = 2
PROVENANCES = ("trained", "smoke", "test_fixture")


class CheckpointError(RuntimeError):
    """Checkpoint missing, unreadable or untrusted."""


class IncompatibleCheckpointError(CheckpointError):
    """Checkpoint was produced for a different task/schema (e.g. the old binary model)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_checkpoint(*, model: SixClassNet, model_cfg: dict, normalization: dict, class_weights: list[float],
                     split_id: str, provenance: str, name: str, input_size: tuple[int, int],
                     training: dict | None = None, extra: dict | None = None) -> dict:
    if provenance not in PROVENANCES:
        raise ValueError(f"provenance must be one of {PROVENANCES}")
    ckpt = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "provenance": provenance,
        "smoke": provenance == "smoke",
        "name": name,
        "model_version": f"{name}-{TASK}-{split_id}" + ("" if provenance == "trained" else f"-{provenance}"),
        "model": {"kind": model_cfg["kind"], "with_decoder": bool(model_cfg["with_decoder"]),
                  "num_classes": NUM_CLASSES, "in_channels": 1, "arch": "resnet18_from_scratch"},
        "input_size": [int(input_size[0]), int(input_size[1])],
        "normalization": {"mean": float(normalization["mean"]), "std": float(normalization["std"])},
        "label_mapping": mapping_for_checkpoint(),
        "class_weights": [float(w) for w in class_weights],
        "split_id": split_id,
        "decision_rule": DECISION_RULE,
        "training": training or {},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        ckpt.update(extra)
    return ckpt


def save_checkpoint(path: Path, ckpt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


@dataclass
class LoadedCheckpoint:
    path: Path
    sha256: str
    meta: dict  # everything except state_dict
    model: SixClassNet
    index_to_source: list[int]


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu",
                    allow_non_trained: bool = False) -> LoadedCheckpoint:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        # weights_only=True: only tensors and plain containers are deserialized (no arbitrary pickle code)
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Could not safely read checkpoint {path.name}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("task") != TASK or raw.get("schema_version") != SCHEMA_VERSION \
            or "threshold" in raw or "label_mapping" not in raw:
        raise IncompatibleCheckpointError(
            f"Checkpoint {path.name} is not a six-class (task={TASK!r}, schema {SCHEMA_VERSION}) checkpoint. "
            "It looks like an older binary visible-defect model (single logit + threshold). "
            "Retrain with the current code: python -m ml.run_all"
        )
    provenance = raw.get("provenance")
    if provenance not in PROVENANCES:
        raise IncompatibleCheckpointError(f"Checkpoint has unknown provenance {provenance!r}")
    if provenance != "trained" and not allow_non_trained:
        raise CheckpointError(
            f"Refusing to load a {provenance!r} checkpoint ({path.name}) as a production model. "
            "Train a real model with: python -m ml.run_all"
        )
    try:
        index_to_source = validate_checkpoint_mapping(raw["label_mapping"])
    except LabelError as exc:
        raise IncompatibleCheckpointError(str(exc)) from exc
    mcfg = raw["model"]
    if int(mcfg.get("num_classes", -1)) != NUM_CLASSES:
        raise IncompatibleCheckpointError(f"Checkpoint has {mcfg.get('num_classes')} outputs, expected {NUM_CLASSES}")
    model = build_model(mcfg)
    try:
        model.load_state_dict(raw["state_dict"], strict=True)
    except RuntimeError as exc:
        raise IncompatibleCheckpointError(f"Weights do not match the six-class architecture: {exc}") from exc
    model.to(device).eval()
    meta = {k: v for k, v in raw.items() if k != "state_dict"}
    return LoadedCheckpoint(path=path, sha256=sha256_file(path), meta=meta, model=model,
                            index_to_source=index_to_source)
