import numpy as np
import pytest
import torch

from ml import labels
from ml.decision import (CORRECT, FAIL, FALSE_FAIL, FALSE_PASS, PASS, WRONG_DEFECT_TYPE, categorize, decide,
                         reference_decision)
from ml.predict import decode_outputs


def test_mapping_class_1_and_6_explicit():
    assert labels.encode(1) == 0
    assert labels.encode(6) == 5
    assert labels.decode(0) == 1
    assert labels.decode(5) == 6


@pytest.mark.parametrize("sid", [1, 2, 3, 4, 5, 6])
def test_round_trip(sid):
    assert labels.decode(labels.encode(sid)) == sid


def test_unknown_ids_rejected():
    for bad in (0, 7, -1):
        with pytest.raises(labels.LabelError):
            labels.encode(bad)
    for bad in (-1, 6, 99):
        with pytest.raises(labels.LabelError):
            labels.decode(bad)


def test_checkpoint_mapping_validation_rejects_shift():
    good = labels.mapping_for_checkpoint()
    assert labels.validate_checkpoint_mapping(good) == [1, 2, 3, 4, 5, 6]
    shifted = dict(good, index_to_source=[2, 3, 4, 5, 6, 1])
    with pytest.raises(labels.LabelError):
        labels.validate_checkpoint_mapping(shifted)


def test_class_names_do_not_invent_defect_names():
    assert [labels.class_name(i) for i in range(1, 7)] == [f"Class {i}" for i in range(1, 7)]


@pytest.mark.parametrize("cid,expected", [(1, FAIL), (2, FAIL), (3, FAIL), (4, FAIL), (5, FAIL), (6, PASS)])
def test_all_six_class_to_decision_mappings(cid, expected):
    d = decide(cid)
    assert d.decision == expected
    assert reference_decision(cid) == expected
    assert d.reason  # explicit reason always present
    assert (d.decision == PASS) == (cid == 6)


def test_wrong_defect_type_still_fails_inspection():
    # true class 3 predicted as class 4: wrong type, but decision FAIL is still correct
    assert categorize(3, 4) == WRONG_DEFECT_TYPE
    assert decide(4).decision == FAIL == reference_decision(3)


def test_defect_incorrectly_passed_as_class_6():
    assert categorize(2, 6) == FALSE_PASS
    assert decide(6).decision == PASS != reference_decision(2)


def test_class_6_flagged_as_defect_is_false_fail():
    assert categorize(6, 1) == FALSE_FAIL
    assert categorize(6, 6) == CORRECT and categorize(5, 5) == CORRECT


def test_decode_outputs_converts_indices_to_source_ids_and_decision_agrees():
    # logits peaked at internal index 0 -> source class 1; index 5 -> source class 6
    for idx, sid in ((0, 1), (5, 6), (2, 3)):
        logits = torch.full((1, 6), -3.0)
        logits[0, idx] = 4.0
        res = decode_outputs(logits, None, [1, 2, 3, 4, 5, 6])[0]
        assert res["predicted_class_id"] == sid
        assert res["inspection_decision"] == (PASS if sid == 6 else FAIL)
        assert set(res["class_scores"]) == {"1", "2", "3", "4", "5", "6"}
        assert res["class_scores"][str(sid)] == pytest.approx(res["predicted_class_score"])
        assert sum(res["class_scores"].values()) == pytest.approx(1.0, abs=1e-5)
        assert np.isclose(max(res["class_scores"].values()), res["predicted_class_score"])
