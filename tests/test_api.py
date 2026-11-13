"""API tests. Prediction-path tests use isolated test-fixture checkpoints (random weights): they
verify plumbing, validation and decision wiring, NOT model quality."""
import json

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.settings import Settings
from tests.fixture_models import make_checkpoint, make_png_bytes


def _client(tmp_path, ckpt=None, allow=True, **kw):
    s = Settings(checkpoint_path=ckpt or (tmp_path / "missing.pt"), results_dir=tmp_path / "results",
                 frontend_dist=tmp_path / "no_dist", allow_non_trained=allow, **kw)
    return TestClient(create_app(s))


def _upload(client, data, name="x.png", ctype="image/png"):
    return client.post("/api/predict", files={"file": (name, data, ctype)})


class Stub(torch.nn.Module):
    """Returns fixed logits peaked at one internal index (to test mapping + decision end to end)."""

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        logits = torch.full((x.shape[0], 6), -2.0)
        logits[:, self.idx] = 3.0
        return {"logits": logits, "mask_logits": None}


def test_server_starts_without_checkpoint_and_reports_it(tmp_path):
    with _client(tmp_path) as c:
        assert c.get("/api/health").json() == {"status": "ok"}
        m = c.get("/api/model").json()
        assert m["ready"] is False and m["reason"] == "checkpoint_missing" and m["model"] is None
        r = _upload(c, make_png_bytes())
        assert r.status_code == 503 and r.json()["detail"]["code"] == "checkpoint_missing"


def test_old_binary_checkpoint_gives_compatibility_error(tmp_path):
    p = tmp_path / "old.pt"
    torch.save({"state_dict": {}, "threshold": 0.5}, p)
    with _client(tmp_path, p) as c:
        m = c.get("/api/model").json()
        assert m["ready"] is False and m["reason"] == "checkpoint_incompatible" and "binary" in m["message"]
        assert _upload(c, make_png_bytes()).status_code == 503


def test_non_trained_checkpoint_refused_when_not_allowed(tmp_path):
    p = make_checkpoint(tmp_path / "f.pt", with_decoder=False)
    with _client(tmp_path, p, allow=False) as c:
        m = c.get("/api/model").json()
        assert m["ready"] is False and m["reason"] == "checkpoint_rejected"


def test_baseline_returns_classification_only(tmp_path):
    p = make_checkpoint(tmp_path / "a.pt", with_decoder=False)
    with _client(tmp_path, p) as c:
        meta = c.get("/api/model").json()
        assert meta["ready"] and meta["model"]["has_mask_output"] is False
        assert meta["model"]["provenance"] == "test_fixture"
        r = _upload(c, make_png_bytes())
        assert r.status_code == 200
        j = r.json()
        for key in ("predicted_class_id", "predicted_class_name", "class_scores", "inspection_decision",
                    "decision_reason", "model_version", "original_width", "original_height",
                    "processing_time_ms", "predicted_class_score"):
            assert key in j
        assert j["predicted_mask"] is None
        assert set(j["class_scores"]) == {"1", "2", "3", "4", "5", "6"}
        assert sum(j["class_scores"].values()) == pytest.approx(1.0, abs=1e-4)
        assert j["original_width"] == j["original_height"] == 480
        assert j["inspection_decision"] == ("PASS" if j["predicted_class_id"] == 6 else "FAIL")
        assert j["class_scores"][str(j["predicted_class_id"])] == pytest.approx(j["predicted_class_score"])


def test_mask_model_returns_aligned_mask(tmp_path):
    import base64
    import io

    from PIL import Image
    p = make_checkpoint(tmp_path / "b.pt", with_decoder=True)
    with _client(tmp_path, p) as c:
        j = _upload(c, make_png_bytes()).json()
        pm = j["predicted_mask"]
        assert pm and (pm["width"], pm["height"]) == (480, 480)
        im = Image.open(io.BytesIO(base64.b64decode(pm["data"])))
        assert im.size == (480, 480) and im.mode == "L"


@pytest.mark.parametrize("idx", range(6))
def test_every_class_maps_to_the_right_source_id_and_decision_through_the_api(tmp_path, idx):
    p = make_checkpoint(tmp_path / "a.pt", with_decoder=False)
    with _client(tmp_path, p) as c:
        c.app.state.state["predictor"].model = Stub(idx)
        j = _upload(c, make_png_bytes()).json()
        sid = idx + 1  # internal index -> original source class ID
        assert j["predicted_class_id"] == sid and j["predicted_class_name"] == f"Class {sid}"
        assert j["inspection_decision"] == ("PASS" if sid == 6 else "FAIL")
        assert max(j["class_scores"], key=j["class_scores"].get) == str(sid)


def test_upload_validation(tmp_path):
    p = make_checkpoint(tmp_path / "a.pt", with_decoder=False)
    with _client(tmp_path, p, max_upload_bytes=200_000) as c:
        assert _upload(c, b"not an image at all", "x.txt", "text/plain").status_code == 415
        assert _upload(c, b"GIF89a....", "x.gif", "image/gif").status_code == 415
        assert _upload(c, b"").status_code == 422
        assert _upload(c, make_png_bytes((100, 100))).json()["detail"]["code"] == "invalid_dimensions"
        big = _upload(c, make_png_bytes((480, 480)) + b"0" * 300_000)
        assert big.status_code == 413
        # a JPEG of the right size is accepted
        assert _upload(c, make_png_bytes(fmt="JPEG"), "x.jpg", "image/jpeg").status_code == 200


def test_experiments_empty_state_and_artifacts(tmp_path):
    with _client(tmp_path) as c:
        j = c.get("/api/experiments").json()
        assert j["available"] is False and "ml.run_all" in j["command"]
        res = tmp_path / "results"
        (res / "gallery").mkdir(parents=True)
        (res / "comparison.json").write_text(json.dumps({"models": {}, "results": {}}))
        (res / "gallery" / "a.png").write_bytes(make_png_bytes((8, 8)))
        (tmp_path / "secret.png").write_bytes(b"x")
        assert c.get("/api/experiments").json()["available"] is True
        assert c.get("/api/experiments/assets/gallery/a.png").status_code == 200
        assert c.get("/api/experiments/assets/../secret.png").status_code in (404,)
        assert c.get("/api/experiments/assets/%2e%2e/secret.png").status_code == 404
