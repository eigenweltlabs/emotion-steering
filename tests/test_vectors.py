"""Test the bundled example vectors are loadable and correctly shaped."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emotion_steering.cli import (
    _auc_tables, _chosen_auc_summary, _metadata_count_summary,
)
from emotion_steering.vectors import load_bundle, save_bundle

EXAMPLE_BUNDLE = Path(__file__).resolve().parent.parent / "examples" / "qwen3-8b-ekman6"


def test_example_bundle_loads():
    if not EXAMPLE_BUNDLE.exists():
        pytest.skip("example bundle not present")
    bundle = load_bundle(EXAMPLE_BUNDLE)
    assert bundle.model == "Qwen/Qwen3-8B"
    assert bundle.hidden == 4096
    assert bundle.chosen_layers == [20, 21, 22]
    assert bundle.emotions == ["anger", "joy", "sadness", "disgust", "fear", "surprise"]
    stacked = bundle.stack_chosen()
    assert stacked.shape == (6, 3, 4096)
    # Norms are non-trivial (not zero, not absurd)
    norms = np.linalg.norm(stacked, axis=-1)
    assert (norms > 5).all()
    assert (norms < 50).all()


def test_example_legacy_metadata_reports_counts_and_auc():
    md = json.loads((EXAMPLE_BUNDLE / "metadata.json").read_text())
    assert _metadata_count_summary(md) == (
        "anger/joy/sadness=7274 / 1819; disgust/fear/surprise=1591 / 398"
    )
    tables = _auc_tables(md, md["emotions"])
    assert len(tables) == 2
    assert [cols for _, cols, _ in tables] == [
        ["anger", "joy", "sadness"],
        ["disgust", "fear", "surprise"],
    ]
    assert _chosen_auc_summary(md, tables) == (
        "anger/joy/sadness=0.829; disgust/fear/surprise=0.846"
    )


def test_save_bundle_roundtrip(tmp_path):
    chosen = {
        "anger": np.random.randn(3, 8).astype(np.float32),
        "joy":   np.random.randn(3, 8).astype(np.float32),
    }
    md = {
        "model": "fake/test", "emotions": ["anger", "joy"],
        "hidden": 8, "chosen_layers": [2, 3, 4],
    }
    save_bundle(tmp_path, chosen=chosen, full_sweep=None, metadata=md)
    bundle = load_bundle(tmp_path)
    assert bundle.emotions == ["anger", "joy"]
    np.testing.assert_array_equal(bundle.chosen["anger"], chosen["anger"])
    np.testing.assert_array_equal(bundle.chosen["joy"], chosen["joy"])
