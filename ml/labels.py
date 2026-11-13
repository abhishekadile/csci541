"""Label handling for the six original Carinthia-S classes.

Source class IDs are 1..6 (as in the dataset CSV). PyTorch cross-entropy needs
indices 0..5, so we map ID -> index for training and index -> ID everywhere a
prediction is evaluated, displayed or returned.

The dataset documentation only states that class 6 images contain no visible
defect. It publishes no descriptive names for classes 1-5, so we display
"Class N" and do not invent physical defect names.
"""
from __future__ import annotations

from typing import Iterable, Sequence

SOURCE_CLASS_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
NUM_CLASSES: int = len(SOURCE_CLASS_IDS)
PASS_CLASS_ID: int = 6  # documented as "no visible defects"

SOURCE_TO_INDEX: dict[int, int] = {sid: i for i, sid in enumerate(SOURCE_CLASS_IDS)}
INDEX_TO_SOURCE: dict[int, int] = {i: sid for sid, i in SOURCE_TO_INDEX.items()}

CLASS_NAMES: dict[int, str] = {sid: f"Class {sid}" for sid in SOURCE_CLASS_IDS}


class LabelError(ValueError):
    pass


def class_name(source_id: int) -> str:
    if source_id not in CLASS_NAMES:
        raise LabelError(f"Unknown source class id {source_id!r}; expected one of {SOURCE_CLASS_IDS}")
    return CLASS_NAMES[source_id]


def encode(source_id: int) -> int:
    """Source ID (1..6) -> internal index (0..5)."""
    try:
        return SOURCE_TO_INDEX[int(source_id)]
    except (KeyError, ValueError, TypeError):
        raise LabelError(f"Unknown source class id {source_id!r}; expected one of {SOURCE_CLASS_IDS}") from None


def decode(index: int) -> int:
    """Internal index (0..5) -> source ID (1..6)."""
    try:
        return INDEX_TO_SOURCE[int(index)]
    except (KeyError, ValueError, TypeError):
        raise LabelError(f"Unknown class index {index!r}; expected 0..{NUM_CLASSES - 1}") from None


def encode_many(source_ids: Iterable[int]) -> list[int]:
    return [encode(s) for s in source_ids]


def decode_many(indices: Iterable[int]) -> list[int]:
    return [decode(i) for i in indices]


def mapping_for_checkpoint() -> dict:
    """JSON/torch-safe representation stored in every checkpoint."""
    return {
        "source_to_index": {str(k): v for k, v in SOURCE_TO_INDEX.items()},
        "index_to_source": [INDEX_TO_SOURCE[i] for i in range(NUM_CLASSES)],
        "pass_class_id": PASS_CLASS_ID,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
    }


def validate_checkpoint_mapping(mapping: dict) -> list[int]:
    """Verify a stored mapping and return its index -> source ID list.

    Raises LabelError if the checkpoint was trained with a different mapping
    than this code base assumes, so predictions can never be silently shifted.
    """
    try:
        index_to_source = [int(x) for x in mapping["index_to_source"]]
        source_to_index = {int(k): int(v) for k, v in mapping["source_to_index"].items()}
        pass_id = int(mapping["pass_class_id"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LabelError(f"Checkpoint label mapping is malformed: {exc}") from exc
    expected = [INDEX_TO_SOURCE[i] for i in range(NUM_CLASSES)]
    if index_to_source != expected or source_to_index != SOURCE_TO_INDEX or pass_id != PASS_CLASS_ID:
        raise LabelError(
            f"Checkpoint label mapping {index_to_source} does not match the expected mapping {expected}."
        )
    return index_to_source


def counts_by_source_id(labels: Sequence[int]) -> dict[int, int]:
    out = {sid: 0 for sid in SOURCE_CLASS_IDS}
    for lab in labels:
        out[int(lab)] += 1
    return out
