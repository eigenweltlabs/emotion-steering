"""Tests that don't need a GPU or network: dataset utility functions."""
from __future__ import annotations

import pytest

from emotion_steering.dataset import (
    EKMAN_MAP, LabeledRecord, _balance, split_train_val,
)


def test_ekman_map_covers_six_categories():
    assert set(EKMAN_MAP) == {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
    # No GoEmotions sub-label is double-mapped
    seen = []
    for subs in EKMAN_MAP.values():
        seen.extend(subs)
    assert len(seen) == len(set(seen)), "GoEmotions sub-label appears in multiple Ekman buckets"


def test_balance_caps_each_class_at_min_count():
    records = (
        [LabeledRecord(text=f"a{i}", label="anger") for i in range(50)]
        + [LabeledRecord(text=f"j{i}", label="joy") for i in range(10)]
        + [LabeledRecord(text=f"s{i}", label="sadness") for i in range(30)]
    )
    out = _balance(records, classes=["anger", "joy", "sadness"], seed=0)
    counts = {c: 0 for c in ["anger", "joy", "sadness"]}
    for r in out:
        counts[r.label] += 1
    assert counts == {"anger": 10, "joy": 10, "sadness": 10}
    assert len(out) == 30


def test_balance_raises_on_missing_class():
    records = [LabeledRecord(text="x", label="anger")]
    with pytest.raises(ValueError):
        _balance(records, classes=["anger", "joy"], seed=0)


def test_split_train_val_is_stratified_and_deterministic():
    records = [
        LabeledRecord(text=f"a{i}", label="anger") for i in range(100)
    ] + [LabeledRecord(text=f"j{i}", label="joy") for i in range(100)]
    tr1, vl1 = split_train_val(records, test_size=0.25, seed=42)
    tr2, vl2 = split_train_val(records, test_size=0.25, seed=42)

    # deterministic
    assert [r.text for r in tr1] == [r.text for r in tr2]
    # stratified: validation is ~25% per class
    val_anger = sum(1 for r in vl1 if r.label == "anger")
    val_joy = sum(1 for r in vl1 if r.label == "joy")
    assert val_anger == 25
    assert val_joy == 25
