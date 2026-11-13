import pytest
import torch

from ml.checkpoint import CheckpointError, IncompatibleCheckpointError, load_checkpoint
from tests.fixture_models import make_checkpoint


def test_fixture_checkpoint_refused_by_default_but_loadable_when_opted_in(tmp_path):
    p = make_checkpoint(tmp_path / "f.pt", with_decoder=False)
    with pytest.raises(CheckpointError, match="Refusing"):
        load_checkpoint(p)
    c = load_checkpoint(p, allow_non_trained=True)
    assert c.meta["provenance"] == "test_fixture" and c.index_to_source == [1, 2, 3, 4, 5, 6]


def test_smoke_checkpoint_refused_by_default(tmp_path):
    p = make_checkpoint(tmp_path / "s.pt", with_decoder=True, provenance="smoke")
    with pytest.raises(CheckpointError, match="smoke"):
        load_checkpoint(p)


def test_trained_provenance_loads_and_carries_everything_needed(tmp_path):
    p = make_checkpoint(tmp_path / "t.pt", with_decoder=True, provenance="trained")
    m = load_checkpoint(p).meta
    for key in ("normalization", "label_mapping", "class_weights", "split_id", "decision_rule", "model",
                "input_size", "model_version", "task"):
        assert key in m
    assert "threshold" not in m  # no binary threshold anywhere


def test_old_binary_checkpoint_rejected_with_clear_error(tmp_path):
    p = tmp_path / "binary.pt"
    torch.save({"state_dict": {"fc.weight": torch.zeros(1, 512)}, "threshold": 0.37, "model": {"kind": "baseline"}}, p)
    with pytest.raises(IncompatibleCheckpointError, match="binary"):
        load_checkpoint(p, allow_non_trained=True)


def test_tampered_label_mapping_rejected(tmp_path):
    p = make_checkpoint(tmp_path / "t.pt", with_decoder=False, provenance="trained")
    raw = torch.load(p, weights_only=True)
    raw["label_mapping"]["index_to_source"] = [6, 2, 3, 4, 5, 1]
    torch.save(raw, p)
    with pytest.raises(IncompatibleCheckpointError, match="mapping"):
        load_checkpoint(p)


def test_pickled_objects_are_not_executed(tmp_path):
    import pickle

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    p = tmp_path / "evil.pt"
    with p.open("wb") as f:
        pickle.dump({"x": Evil()}, f)
    with pytest.raises(CheckpointError):
        load_checkpoint(p, allow_non_trained=True)


def test_missing_checkpoint(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")

from ml.config import weights_dir, run_dir  # noqa: E402


def test_weights_and_logs_are_separate_folders(monkeypatch, tmp_path):
    monkeypatch.delenv("WEIGHTS_DIR", raising=False)
    cfg = {"name": "model_x"}
    assert weights_dir(cfg).parts[-2:] == ("weight", "model_x")
    assert "weight" not in run_dir(cfg).parts
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    assert weights_dir(cfg) == tmp_path / "model_x"
