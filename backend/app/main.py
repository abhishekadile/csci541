"""FastAPI inference service for the Wafer Defect Inspection project.

One configured checkpoint is loaded once at startup. If it is missing or unusable the server
still starts, /api/model reports exactly why, and /api/predict answers 503. No random scores,
no fake masks, no invented metrics.
"""
from __future__ import annotations

import base64
import io
import json
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from ml.checkpoint import CheckpointError, IncompatibleCheckpointError
from ml.decision import DECISION_RULE
from ml.labels import CLASS_NAMES, SOURCE_CLASS_IDS
from ml.predict import Predictor

from .settings import Settings

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png"}
MAX_PIXELS = 4096 * 4096
MIN_SIDE = 32
GENERATE_COMMAND = "python -m ml.run_all"
SCOPE = "visual_inspection_of_imaged_site"


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def load_predictor(settings: Settings) -> tuple[Predictor | None, dict]:
    """Returns (predictor or None, readiness info)."""
    path = settings.checkpoint_path
    if not path.is_file():
        return None, {"ready": False, "reason": "checkpoint_missing",
                      "message": f"No checkpoint at the configured path {path}. Train a model first "
                                 f"({GENERATE_COMMAND}) or set MODEL_CHECKPOINT_PATH."}
    try:
        predictor = Predictor.from_path(path, settings.device, settings.allow_non_trained)
    except IncompatibleCheckpointError as exc:
        return None, {"ready": False, "reason": "checkpoint_incompatible", "message": str(exc)}
    except CheckpointError as exc:
        return None, {"ready": False, "reason": "checkpoint_rejected", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface any load failure as a readiness reason
        return None, {"ready": False, "reason": "checkpoint_load_failed", "message": f"{type(exc).__name__}: {exc}"}
    return predictor, {"ready": True, "reason": None, "message": "Model loaded."}


def model_metadata(settings: Settings, predictor: Predictor | None, status: dict) -> dict:
    out = {**status, "classes": [{"id": s, "name": CLASS_NAMES[s]} for s in SOURCE_CLASS_IDS],
           "decision_rule": DECISION_RULE, "scope": SCOPE, "device": settings.device, "model": None}
    if predictor is not None:
        m = predictor.ckpt.meta
        out["model"] = {
            "model_version": predictor.model_version, "name": m["name"], "kind": m["model"]["kind"],
            "has_mask_output": predictor.has_mask_output, "input_size": list(predictor.input_size),
            "split_id": m["split_id"], "provenance": m["provenance"], "epoch": m.get("epoch"),
            "checkpoint_sha256": predictor.ckpt.sha256, "task": m["task"],
            "val_macro_f1": m.get("val_macro_f1"),
        }
    return out


def encode_mask_png(mask: np.ndarray) -> dict:
    buf = io.BytesIO()
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(buf, format="PNG")
    h, w = mask.shape
    return {"encoding": "png_base64", "width": w, "height": h, "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            "defect_pixels": int(mask.sum()), "meaning": "255 = predicted defect, aligned to the original image pixels"}


def read_upload(data: bytes, predictor: Predictor, strict_size: bool = True) -> np.ndarray:
    """Validate content and dimensions; returns a grayscale uint8 array at native resolution."""
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in ALLOWED_FORMATS:
                raise _err(415, "unsupported_file_type", f"Unsupported image format {fmt!r}. Upload a JPEG or PNG SEM image.")
            w, h = im.size
            if w < MIN_SIDE or h < MIN_SIDE or w * h > MAX_PIXELS:
                raise _err(422, "invalid_dimensions", f"Image size {w}x{h} is outside the accepted range.")
            if strict_size and (h, w) != tuple(predictor.input_size):
                eh, ew = predictor.input_size
                raise _err(422, "invalid_dimensions",
                           f"Image is {w}x{h}; this model was trained on {ew}x{eh} images and the service never "
                           f"resizes uploads. Upload a native {ew}x{eh} SEM image.")
            im.load()
            return np.asarray(im.convert("L"), dtype=np.uint8)
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise _err(415, "unsupported_file_type", "The file is not a readable JPEG or PNG image.") from None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state: dict = {"predictor": None, "status": {"ready": False, "reason": "not_loaded", "message": "Starting."}}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["predictor"], state["status"] = load_predictor(settings)
        yield

    app = FastAPI(title="Wafer Defect Inspection API", version="1.0.0", lifespan=lifespan)
    app.state.state = state
    app.state.settings = settings

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/model")
    def model_info() -> dict:
        return model_metadata(settings, state["predictor"], state["status"])

    @app.post("/api/predict")
    def predict(file: UploadFile = File(...)) -> dict:
        predictor: Predictor | None = state["predictor"]
        if predictor is None:
            raise _err(503, state["status"]["reason"], state["status"]["message"])
        t0 = time.perf_counter()
        data = file.file.read(settings.max_upload_bytes + 1)  # never hold more than the limit + 1 byte
        if len(data) > settings.max_upload_bytes:
            raise _err(413, "file_too_large", f"File exceeds the {settings.max_upload_bytes} byte limit.")
        if not data:
            raise _err(422, "empty_file", "The uploaded file is empty.")
        img = read_upload(data, predictor)  # raises 415 / 422; the upload stays in memory only
        res = predictor.predict_u8(img)
        mask = res.pop("predicted_mask")
        h, w = img.shape
        return {
            "predicted_class_id": res["predicted_class_id"],
            "predicted_class_name": res["predicted_class_name"],
            "predicted_class_score": res["predicted_class_score"],
            "class_scores": res["class_scores"],  # keyed by original source class IDs "1".."6"
            "inspection_decision": res["inspection_decision"],
            "decision_reason": res["decision_reason"],
            "scope": SCOPE,
            "model_version": res["model_version"],
            "model_kind": predictor.ckpt.meta["model"]["kind"],
            "original_width": w, "original_height": h,
            "predicted_mask": encode_mask_png(mask) if mask is not None else None,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    @app.get("/api/experiments")
    def experiments() -> JSONResponse:
        path = settings.results_dir / "comparison.json"
        if not path.is_file():
            return JSONResponse({"available": False, "command": GENERATE_COMMAND,
                                 "message": "No evaluation artifacts found. Train and evaluate both models to generate them."})
        return JSONResponse({"available": True, "command": GENERATE_COMMAND, **json.loads(path.read_text("utf-8"))})

    @app.get("/api/experiments/assets/{asset_path:path}")
    def experiment_asset(asset_path: str) -> FileResponse:
        base = settings.results_dir.resolve()
        target = (base / asset_path).resolve()
        if base not in target.parents or target.suffix.lower() != ".png" or not target.is_file():
            raise _err(404, "asset_not_found", "Asset not found.")
        return FileResponse(target, media_type="image/png")

    dist = settings.frontend_dist
    if dist.is_dir() and (dist / "index.html").is_file():
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise _err(404, "not_found", "Unknown API route.")
            target = (dist / full_path).resolve()
            if full_path and dist.resolve() in target.parents and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
