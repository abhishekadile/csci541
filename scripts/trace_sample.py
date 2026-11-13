"""Trace ONE real labeled dataset sample through the whole chain:

    CSV label -> internal index -> model logits -> decoded source ID -> PASS/FAIL
              -> API JSON response -> what the frontend renders

    python scripts/trace_sample.py --class-id 6 --checkpoint weight/model_b/best.pt
    python scripts/trace_sample.py --class-id 1 --checkpoint <file> --allow-non-trained   # pipeline checks only

The output states the checkpoint provenance so it is clear whether trained weights were used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import create_app  # noqa: E402
from backend.app.settings import Settings  # noqa: E402
from ml.checkpoint import load_checkpoint  # noqa: E402
from ml.config import get_data_root, load_config  # noqa: E402
from ml.data import normalize_u8  # noqa: E402
from ml.dataset_index import load_image_u8, read_records  # noqa: E402
from ml.decision import decide, reference_decision  # noqa: E402
from ml.labels import decode, encode  # noqa: E402
from ml.split import load_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--allow-non-trained", action="store_true")
    ap.add_argument("--index", type=int, default=0, help="which sample of that class in the split")
    a = ap.parse_args()

    cfg = load_config()
    manifest = [r for r in load_manifest(cfg) if r["split"] == a.split and r["label"] == a.class_id]
    if not manifest:
        raise SystemExit(f"No class {a.class_id} sample in split {a.split}")
    row = manifest[a.index]
    rec = {r.filename: r for r in read_records(get_data_root())}[row["filename"]]

    print("1. DATASET ROW")
    print(f"   file={rec.image_path.name} split={a.split} source label={row['label']} "
          f"(reference decision from true label: {reference_decision(row['label'])})")

    idx = encode(row["label"])
    print("2. LABEL ENCODING")
    print(f"   source ID {row['label']} -> internal index {idx} -> decoded back {decode(idx)}")
    assert decode(idx) == row["label"]

    ck = load_checkpoint(a.checkpoint, "cpu", a.allow_non_trained)
    print(f"   checkpoint: {ck.path.name} provenance={ck.meta['provenance']} version={ck.meta['model_version']} "
          f"epoch={ck.meta.get('epoch')}")
    print(f"   checkpoint index->source mapping: {ck.index_to_source}")

    img = load_image_u8(rec.image_path)
    with torch.no_grad():
        x = normalize_u8(img, ck.meta["normalization"]["mean"], ck.meta["normalization"]["std"]).unsqueeze(0)
        out = ck.model(x)
    probs = torch.softmax(out["logits"], 1)[0].numpy()
    top = int(np.argmax(probs))
    print("3. MODEL OUTPUT DECODING")
    print(f"   softmax by internal index: {np.round(probs, 4).tolist()}")
    print(f"   argmax internal index {top} -> source class {ck.index_to_source[top]}")
    d = decide(ck.index_to_source[top])
    print(f"   decision from ml.decision.decide: {d.decision} ({d.reason})")

    print("4. API RESPONSE (POST /api/predict with the original file bytes)")
    settings = Settings(checkpoint_path=Path(a.checkpoint).resolve(), allow_non_trained=a.allow_non_trained,
                        frontend_dist=Path("nonexistent"))
    with TestClient(create_app(settings)) as c:
        r = c.post("/api/predict", files={"file": (rec.image_path.name, rec.image_path.read_bytes(), "image/jpeg")})
        r.raise_for_status()
        j = r.json()
    summary = {k: j[k] for k in ("predicted_class_id", "predicted_class_name", "predicted_class_score",
                                 "inspection_decision", "decision_reason", "model_version", "processing_time_ms")}
    print(f"   {summary}")
    print(f"   class_scores keys: {sorted(j['class_scores'])}; mask returned: {j['predicted_mask'] is not None}")
    assert j["predicted_class_id"] == ck.index_to_source[top], "API class differs from direct model decoding"
    assert j["inspection_decision"] == d.decision, "API decision differs from decide()"
    assert max(j["class_scores"], key=j["class_scores"].get) == str(j["predicted_class_id"])

    print("5. WHAT THE FRONTEND RENDERS (fields it reads from the API; it computes no decision itself)")
    print(f"   Visual inspection: {j['inspection_decision']}  |  {j['predicted_class_name']} "
          f"(source ID {j['predicted_class_id']})  |  softmax {100 * j['predicted_class_score']:.1f}%")
    print("   OK: API output matches direct decoding and the shared decision function.")
    print(f"   Reference vs predicted decision: {reference_decision(row['label'])} vs {j['inspection_decision']}"
          + ("  (this sample is classified incorrectly by these weights)" if j["predicted_class_id"] != row["label"] else ""))


if __name__ == "__main__":
    main()
