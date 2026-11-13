import numpy as np
import pytest

from ml import metrics as M


def test_confusion_matrix_orientation_rows_true_cols_pred():
    cm = M.confusion_matrix([1, 6, 6], [6, 6, 1])
    assert cm[0, 5] == 1  # true 1 predicted 6
    assert cm[5, 5] == 1 and cm[5, 0] == 1
    assert cm.sum() == 3


def test_macro_f1_ignores_classes_without_support_and_reports_unavailable():
    y_true = [3, 3, 3, 6]
    y_pred = [3, 3, 4, 6]
    rep = M.classification_report(M.confusion_matrix(y_true, y_pred))
    assert rep["macro_f1_classes"] == [3, 6]
    c1 = rep["per_class"][0]
    assert c1["support"] == 0 and c1["recall"] is None and c1["f1"] is None
    assert "unavailable" in c1["recall_text"]
    f1_3 = 2 * 2 / (2 * 2 + 0 + 1)
    assert rep["macro_f1"] == pytest.approx((f1_3 + 1.0) / 2)
    assert rep["per_class"][2]["recall_text"] == "2 of 3 correct"


def test_present_but_never_predicted_class_counts_as_zero_f1():
    rep = M.classification_report(M.confusion_matrix([5, 6], [6, 6]))
    assert rep["per_class"][4]["f1"] == 0.0
    assert rep["macro_f1"] == pytest.approx((0.0 + 2 * 1 / (2 * 1 + 1)) / 2)


def test_inspection_report_separates_three_error_types():
    y_true = [1, 2, 3, 3, 6, 6, 6]
    y_pred = [6, 2, 4, 3, 1, 6, 6]  # false pass (1->6), wrong type (3->4), false fail (6->1)
    r = M.inspection_report(y_true, y_pred)
    assert (r["false_pass"]["num"], r["false_pass"]["den"]) == (1, 4)
    assert (r["wrong_defect_type"]["num"], r["wrong_defect_type"]["den"]) == (1, 4)
    assert (r["false_fail"]["num"], r["false_fail"]["den"]) == (1, 3)
    assert (r["defects_flagged_fail"]["num"], r["defects_flagged_fail"]["den"]) == (3, 4)


def test_rates_none_when_denominator_zero():
    r = M.inspection_report([1, 2], [1, 2])
    assert r["false_fail"]["rate"] is None and r["false_fail"]["den"] == 0


def test_majority_baseline_uses_train_majority():
    b = M.majority_baseline([3, 3, 3, 6], [3, 6, 6])
    assert b["predicts_class_id"] == 3
    assert b["accuracy"]["num"] == 1


def test_size_group_report_counts_flagged_and_exact():
    rep = M.size_group_report([0, 0, 0], [3, 4, 3], [3, 3, 6], ["small", "small", "large"])
    s = rep["small"]
    assert s["flagged_fail"]["num"] == 2 and s["exact_class_correct"]["num"] == 1
    assert rep["large"]["missed_as_pass"]["num"] == 1 and rep["medium"]["n"] == 0


def test_mask_report_handles_empty_masks_separately():
    a = M.mask_overlap(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [0, 0]]))
    e = M.mask_overlap(np.array([[0, 0], [0, 1]]), np.zeros((2, 2)))
    rep = M.mask_report([a, e], [3, 6])
    assert rep["images_with_expert_mask"] == 1
    assert rep["mean_dice"] == pytest.approx(2 * 1 / (2 + 1))
    assert rep["mean_iou"] == pytest.approx(1 / 2)
    assert rep["empty_expert_mask_images"] == 1
    assert rep["empty_mask_images_with_any_predicted_pixels"]["num"] == 1
