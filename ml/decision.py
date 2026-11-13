"""The single project rule that turns a predicted class into a visual-inspection decision.

Used by evaluation, CLI inference and the API. The frontend only displays what the
API returns.

Project rule (applied to the ORIGINAL source class ID, not the internal index):
    predicted class 6        -> PASS  (no visible defect)
    predicted class 1,2,3,4,5 -> FAIL (visible defect)

Scope: this is a visual inspection of one imaged SEM site. It says nothing about
electrical die yield or the disposition of a whole wafer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .labels import PASS_CLASS_ID, SOURCE_CLASS_IDS, class_name

PASS = "PASS"
FAIL = "FAIL"
DECISION_RULE = "visual_inspection_v1: class 6 -> PASS (no visible defect); classes 1-5 -> FAIL (visible defect)"

# Outcome categories for (true class, predicted class)
CORRECT = "correct"
WRONG_DEFECT_TYPE = "wrong_defect_type"  # defect predicted as another defect class: decision still FAIL
FALSE_PASS = "false_pass"  # actual defect (1-5) predicted as class 6
FALSE_FAIL = "false_fail"  # actual class 6 predicted as a defect class


@dataclass(frozen=True)
class InspectionDecision:
    decision: str
    reason: str

    def as_dict(self) -> dict:
        return {"inspection_decision": self.decision, "decision_reason": self.reason}


def _check(class_id: int) -> int:
    class_id = int(class_id)
    if class_id not in SOURCE_CLASS_IDS:
        raise ValueError(f"Unknown source class id {class_id}; expected one of {SOURCE_CLASS_IDS}")
    return class_id


def decide(predicted_class_id: int) -> InspectionDecision:
    """Decision derived from a predicted source class ID."""
    cid = _check(predicted_class_id)
    if cid == PASS_CLASS_ID:
        return InspectionDecision(PASS, f"Predicted {class_name(cid)}: no visible defect in this imaged site.")
    return InspectionDecision(FAIL, f"Predicted {class_name(cid)}: visible defect in this imaged site.")


def reference_decision(true_class_id: int) -> str:
    """Reference PASS/FAIL derived from the true source label with the same mapping."""
    return PASS if _check(true_class_id) == PASS_CLASS_ID else FAIL


def categorize(true_class_id: int, predicted_class_id: int) -> str:
    t, p = _check(true_class_id), _check(predicted_class_id)
    if t == p:
        return CORRECT
    if t == PASS_CLASS_ID:
        return FALSE_FAIL
    if p == PASS_CLASS_ID:
        return FALSE_PASS
    return WRONG_DEFECT_TYPE
