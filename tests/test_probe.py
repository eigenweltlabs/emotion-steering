"""Probe smoke test on synthetic (CPU-only) data."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from emotion_steering.probe import best_window, probe_one


def test_best_window_picks_argmax():
    auc = np.array([
        [0.5, 0.5],
        [0.6, 0.6],
        [0.9, 0.9],   # this row
        [0.9, 0.9],   # and this
        [0.9, 0.9],   # and this -> window at start_idx=2 wins
        [0.4, 0.4],
    ])
    start, mean = best_window(auc, window=3)
    assert start == 2
    assert mean == pytest.approx(0.9)


def test_best_window_respects_layer_id_gaps():
    auc = np.array([
        [0.9, 0.9],
        [0.9, 0.9],
        [0.9, 0.9],
        [0.6, 0.6],
        [0.6, 0.6],
        [0.6, 0.6],
    ])

    start, mean = best_window(auc, window=3, layer_ids=[4, 20, 21, 22, 32, 33])

    assert start == 1
    assert mean == pytest.approx(0.8)


def test_best_window_rejects_missing_contiguous_window():
    auc = np.array([
        [0.9, 0.9],
        [0.8, 0.8],
        [0.7, 0.7],
    ])

    with pytest.raises(ValueError, match="no contiguous"):
        best_window(auc, window=2, layer_ids=[4, 20, 32])


def test_probe_one_separates_clearly_separable_data():
    """Linearly separable Gaussians should give AUC ~ 1.0."""
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    rng = np.random.default_rng(0)
    n, d = 200, 16
    pos = rng.normal(loc=2.0, size=(n // 2, d)).astype(np.float32)
    neg = rng.normal(loc=-2.0, size=(n // 2, d)).astype(np.float32)
    train_x = np.concatenate([pos, neg], axis=0)
    train_y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.float32)

    pos_v = rng.normal(loc=2.0, size=(50, d)).astype(np.float32)
    neg_v = rng.normal(loc=-2.0, size=(50, d)).astype(np.float32)
    val_x = np.concatenate([pos_v, neg_v], axis=0)
    val_y = np.concatenate([np.ones(50), np.zeros(50)]).astype(np.float32)

    auc = probe_one(train_x, train_y, val_x, val_y, device=device)
    assert auc > 0.95
