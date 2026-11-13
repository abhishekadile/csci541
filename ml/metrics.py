"""Metrics. Pure numpy; every function takes ORIGINAL source class IDs (1..6).

Conventions
* Confusion matrix rows = true class, columns = predicted class, both in order 1..6.
* A class with zero true examples in the split has recall/F1 "unavailable" (None).
* Macro-F1 = unweighted mean of F1 over the classes that have >= 1 true example in the split.
  F1 = 2TP / (2TP + FP + FN), so a present class that is never predicted counts as F1 = 0.
* Rates keep their raw numerator and denominator and are None when the denominator is 0.
"""
from __future__ import annotations

import numpy as np

from .decision import (CORRECT, FALSE_FAIL, FALSE_PASS, WRONG_DEFECT_TYPE, categorize, reference_decision)
from .labels import PASS_CLASS_ID, SOURCE_CLASS_IDS, class_name


def rate(num: int, den: int) -> dict:
    return {"num": int(num), "den": int(den), "rate": (num / den) if den > 0 else None}


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    cm = np.zeros((len(SOURCE_CLASS_IDS), len(SOURCE_CLASS_IDS)), dtype=np.int64)
    idx = {sid: i for i, sid in enumerate(SOURCE_CLASS_IDS)}
    for t, p in zip(y_true, y_pred):
        cm[idx[int(t)], idx[int(p)]] += 1
    return cm


def classification_report(cm: np.ndarray) -> dict:
    per_class = []
    f1s = []
    for i, sid in enumerate(SOURCE_CLASS_IDS):
        support = int(cm[i].sum())
        tp = int(cm[i, i])
        predicted = int(cm[:, i].sum())
        fp, fn = predicted - tp, support - tp
        entry = {
            "class_id": sid, "class_name": class_name(sid), "support": support, "correct": tp,
            "predicted_count": predicted,
            "recall": (tp / support) if support else None,
            "precision": (tp / predicted) if predicted else None,
            "f1": (2 * tp / (2 * tp + fp + fn)) if support else None,
            "recall_text": f"{tp} of {support} correct" if support else "unavailable (no examples in this split)",
        }
        if support:
            f1s.append(entry["f1"])
        per_class.append(entry)
    n = int(cm.sum())
    return {
        "per_class": per_class,
        "accuracy": rate(int(np.trace(cm)), n),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_f1_classes": [e["class_id"] for e in per_class if e["support"]],
        "macro_f1_note": "unweighted mean of per-class F1 over classes with >=1 true example in this split",
        "macro_recall": float(np.mean([e["recall"] for e in per_class if e["support"]])) if n else None,
        "n": n,
    }


def majority_baseline(train_labels, y_true) -> dict:
    """Always predict the most frequent TRAIN class."""
    counts = {sid: int((np.asarray(train_labels) == sid).sum()) for sid in SOURCE_CLASS_IDS}
    maj = max(counts, key=counts.get)
    cm = confusion_matrix(y_true, [maj] * len(y_true))
    rep = classification_report(cm)
    return {"predicts_class_id": maj, "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
            "inspection": inspection_report(y_true, [maj] * len(y_true))}


def inspection_report(y_true, y_pred) -> dict:
    y_true, y_pred = list(map(int, y_true)), list(map(int, y_pred))
    cats = [categorize(t, p) for t, p in zip(y_true, y_pred)]
    n_defect = sum(t != PASS_CLASS_ID for t in y_true)
    n_pass = len(y_true) - n_defect
    flagged_fail = sum(t != PASS_CLASS_ID and p != PASS_CLASS_ID for t, p in zip(y_true, y_pred))
    n_fp = cats.count(FALSE_PASS)
    n_ff = cats.count(FALSE_FAIL)
    n_wt = cats.count(WRONG_DEFECT_TYPE)
    ref_ok = sum(reference_decision(t) == ("PASS" if p == PASS_CLASS_ID else "FAIL") for t, p in zip(y_true, y_pred))
    return {
        "n_actual_defect": n_defect, "n_actual_pass": n_pass,
        "false_pass": {**rate(n_fp, n_defect), "definition": "actual classes 1-5 predicted as class 6 / all actual defect images"},
        "false_fail": {**rate(n_ff, n_pass), "definition": "actual class 6 predicted as class 1-5 / all actual class 6 images"},
        "wrong_defect_type": {**rate(n_wt, n_defect),
                              "definition": "actual defect predicted as a DIFFERENT defect class (decision still FAIL) / all actual defect images"},
        "defects_flagged_fail": rate(flagged_fail, n_defect),
        "decision_accuracy": rate(ref_ok, len(y_true)),
    }


def size_group_report(areas, y_true, y_pred, groups) -> dict:
    """`groups` = size-group name per image (positive images only are expected)."""
    out = {}
    for g in ("small", "medium", "large"):
        sel = [i for i, gg in enumerate(groups) if gg == g]
        n = len(sel)
        flagged = sum(int(y_pred[i]) != PASS_CLASS_ID for i in sel)
        exact = sum(int(y_pred[i]) == int(y_true[i]) for i in sel)
        out[g] = {"n": n, "flagged_fail": rate(flagged, n), "missed_as_pass": rate(n - flagged, n),
                  "exact_class_correct": rate(exact, n)}
    return out


def mask_overlap(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return {"inter": inter, "pred": int(p.sum()), "gt": int(g.sum()), "union": union}


def mask_report(items: list[dict], true_ids: list[int]) -> dict:
    """items = outputs of mask_overlap per image. Empty ground-truth masks are reported separately."""
    pos = [it for it, t in zip(items, true_ids) if it["gt"] > 0]
    empty = [it for it, t in zip(items, true_ids) if it["gt"] == 0]
    dice = [2 * it["inter"] / (it["pred"] + it["gt"]) if (it["pred"] + it["gt"]) else 1.0 for it in pos]
    iou = [it["inter"] / it["union"] if it["union"] else 1.0 for it in pos]
    return {
        "images_with_expert_mask": len(pos),
        "mean_dice": float(np.mean(dice)) if dice else None,
        "mean_iou": float(np.mean(iou)) if iou else None,
        "empty_expert_mask_images": len(empty),
        "empty_mask_images_with_any_predicted_pixels": rate(sum(it["pred"] > 0 for it in empty), len(empty)),
        "note": "Dice/IoU averaged over images whose expert mask is non-empty; images with an empty expert mask are reported separately.",
    }
