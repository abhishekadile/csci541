import json

import numpy as np
import pytest
import torch
from PIL import Image

from ml.config import artifacts_dir, load_config
from ml.data import WaferDataset, augment_pair, pad_collate
from ml.dataset_index import Record, load_mask_binary
from ml.losses import compute_class_weights, segmentation_loss
from ml.models import build_model
from ml.split import SPLITS, _targets, size_group


def test_augment_moves_image_and_mask_identically():
    h = w = 64
    img = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    img[10, 20] = 1.0
    mask[10, 20] = 1
    cfg = {"vflip_p": 0.5, "hflip_p": 0.5, "max_translate_px": 7, "brightness": 0.0, "contrast": 0.0}
    for seed in range(40):
        fi, fm = augment_pair(img, mask, cfg, np.random.default_rng(seed))
        assert fi.shape == (h, w) and fm.shape == (h, w)
        assert fm.dtype == np.uint8 and set(np.unique(fm)) <= {0, 1}  # no interpolation of masks
        assert np.argwhere(fi == fi.max())[0].tolist() == np.argwhere(fm == 1)[0].tolist()


def test_dataset_keeps_native_resolution_and_no_resize(tmp_path):
    arr = (np.arange(480 * 480).reshape(480, 480) % 251).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(tmp_path / "a.png")
    m = np.zeros((480, 480), np.uint8)
    m[100:110, 100:110] = 255
    Image.fromarray(m, mode="L").save(tmp_path / "m.png")
    rec = Record("a", 3, tmp_path / "a.png", tmp_path / "m.png")
    item = WaferDataset([rec], 0.4, 0.05)[0]
    assert tuple(item["image"].shape) == (1, 480, 480)
    assert tuple(item["mask"].shape) == (1, 480, 480)
    assert item["label_index"] == 2 and item["source_id"] == 3
    assert int(item["mask"].sum()) == 100


def test_rgba_mask_files_are_binarised_to_single_channel(tmp_path):
    Image.new("RGBA", (16, 16), (0, 0, 0, 255)).save(tmp_path / "m.png")
    assert load_mask_binary(tmp_path / "m.png").sum() == 0


def test_pad_collate_pads_only_when_sizes_differ():
    b = [{"image": torch.zeros(1, 8, 8), "mask": torch.zeros(1, 8, 8), "has_mask": True, "label_index": 0,
          "source_id": 1, "filename": "x"},
         {"image": torch.zeros(1, 8, 6), "mask": torch.zeros(1, 8, 6), "has_mask": True, "label_index": 1,
          "source_id": 2, "filename": "y"}]
    out = pad_collate(b)
    assert out["image"].shape == (2, 1, 8, 8) and out["valid_hw"].tolist() == [[8, 8], [8, 6]]


@pytest.mark.parametrize("with_decoder", [False, True])
def test_model_outputs(with_decoder):
    model = build_model({"kind": "x", "with_decoder": with_decoder}).eval()
    x = torch.randn(2, 1, 96, 80)  # also works at non-square, non-480 sizes (fully convolutional)
    out = model(x)
    assert out["logits"].shape == (2, 6)
    if with_decoder:
        assert out["mask_logits"].shape == (2, 1, 96, 80)
    else:
        assert out["mask_logits"] is None
    assert model.conv1.in_channels == 1


def test_models_share_encoder_architecture_and_start_from_random_init():
    a = build_model({"kind": "a", "with_decoder": False})
    b = build_model({"kind": "b", "with_decoder": True})
    enc_keys = [k for k in a.state_dict() if k in b.state_dict()]
    assert len(enc_keys) == len(a.state_dict())  # B contains every A tensor name and shape
    assert all(a.state_dict()[k].shape == b.state_dict()[k].shape for k in enc_keys)
    assert not any(p.requires_grad is False for p in a.parameters())


def test_class_weights_use_only_given_counts_and_have_unit_mean():
    idx = [0] * 39 + [1] * 6 + [2] * 2806 + [3] * 203 + [4] * 2 + [5] * 159
    for mode in ("sqrt_inverse", "balanced", "none"):
        w = compute_class_weights(idx, mode)
        counts = np.bincount(idx, minlength=6)
        assert (np.array(w) * counts).sum() / counts.sum() == pytest.approx(1.0)
    w = compute_class_weights(idx, "sqrt_inverse")
    assert w[4] > w[0] > w[2]  # rarer -> larger weight
    with pytest.raises(ValueError):
        compute_class_weights([0, 1, 2], "balanced")  # a class missing from train


def test_segmentation_loss_finite_on_empty_masks():
    loss = segmentation_loss(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
    assert torch.isfinite(loss)


def test_split_targets_and_size_group_boundaries():
    assert _targets(4, 0.15, 0.15) == (1, 1)
    assert _targets(2, 0.15, 0.15) == (0, 1)
    assert _targets(1, 0.15, 0.15) == (0, 0)
    c = {"small_max": 0.01, "medium_max": 0.02}
    assert [size_group(a, c) for a in (0.005, 0.01, 0.015, 0.02, 0.5)] == ["small", "small", "medium", "medium", "large"]


def test_real_manifest_no_leakage_between_splits_if_present():
    cfg = load_config()
    path = artifacts_dir() / "splits" / cfg["split"]["id"] / "manifest.csv"
    if not path.is_file():
        pytest.skip("manifest not generated yet (python -m ml.split)")
    import csv
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    groups: dict[str, set] = {}
    for r in rows:
        groups.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in groups.values()), "a duplicate group straddles splits"
    assert {r["split"] for r in rows} == set(SPLITS)
    assert len({r["filename"] for r in rows}) == len(rows)
    report = json.loads((path.parent / "split_report.json").read_text())
    for c in "123456":
        assert sum(report["counts_per_class"][s][c] for s in SPLITS) == sum(1 for r in rows if r["label"] == c)
