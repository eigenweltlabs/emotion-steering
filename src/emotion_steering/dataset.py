"""Datasets for contrastive emotion-vector extraction.

Default: GoEmotions (Demszky et al. 2020) aggregated to Ekman 6 categories.
Users can supply a custom (text, label) iterable to extract any contrastive set.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

# Demszky 2020 Table 4: 27 GoEmotions labels -> Ekman 6.
EKMAN_MAP: dict[str, list[str]] = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief",
    ],
    "sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "surprise": ["confusion", "curiosity", "realization", "surprise"],
    # "neutral" is excluded — no contrastive direction.
}


@dataclass
class LabeledRecord:
    text: str
    label: str


def load_goemotions_ekman(
    target_emotions: list[str],
    seed: int = 42,
    ekman_map: dict[str, list[str]] | None = None,
) -> list[LabeledRecord]:
    """Load GoEmotions, filter to records that map to a single target Ekman label,
    then balance classes (cap each at the smallest class size).

    Returns a shuffled, class-balanced list of LabeledRecord.
    """
    from datasets import load_dataset

    mapping = ekman_map or EKMAN_MAP
    invalid = set(target_emotions) - set(mapping)
    if invalid:
        raise ValueError(
            f"target_emotions {sorted(invalid)} not in ekman_map keys {sorted(mapping)}"
        )

    ds = load_dataset("go_emotions", "simplified")
    go_labels = ds["train"].features["labels"].feature.names
    go_to_ekman = {sub: ek for ek, subs in mapping.items() for sub in subs}

    def single_ekman(label_idxs: list[int]) -> str | None:
        eks = {go_to_ekman[go_labels[i]] for i in label_idxs if go_labels[i] in go_to_ekman}
        return eks.pop() if len(eks) == 1 else None

    records: list[LabeledRecord] = []
    for split in ("train", "validation", "test"):
        for r in ds[split]:
            ek = single_ekman(r["labels"])
            if ek in target_emotions:
                records.append(LabeledRecord(text=r["text"], label=ek))

    return _balance(records, target_emotions, seed=seed)


def _balance(
    records: list[LabeledRecord],
    classes: list[str],
    seed: int,
) -> list[LabeledRecord]:
    counts = Counter(r.label for r in records)
    if not all(c in counts for c in classes):
        missing = [c for c in classes if c not in counts]
        raise ValueError(f"no records found for classes: {missing}")
    target_n = min(counts[c] for c in classes)
    rng = random.Random(seed)
    out: list[LabeledRecord] = []
    for c in classes:
        bucket = [r for r in records if r.label == c]
        rng.shuffle(bucket)
        out.extend(bucket[:target_n])
    rng.shuffle(out)
    return out


def split_train_val(
    records: list[LabeledRecord],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[LabeledRecord], list[LabeledRecord]]:
    """Stratified split by label."""
    from sklearn.model_selection import train_test_split

    texts = [r.text for r in records]
    labels = [r.label for r in records]
    tr_t, vl_t, tr_l, vl_l = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed,
    )
    return (
        [LabeledRecord(text=text, label=label) for text, label in zip(tr_t, tr_l)],
        [LabeledRecord(text=text, label=label) for text, label in zip(vl_t, vl_l)],
    )
