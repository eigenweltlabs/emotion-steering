"""Pure-numpy tests for the contrastive vector math + layer-search defaults."""
from __future__ import annotations

import numpy as np
import pytest

from emotion_steering.extract import (
    build_vectors, default_search_layers, layer_preset_layers, parse_layer_selector,
)


def test_default_search_layers_picks_middle_window():
    layers = default_search_layers(36)
    # Should land roughly in the 16-27 range (44% to 78% of depth)
    assert layers[0] >= 15
    assert layers[-1] <= 28
    assert all(layers[i + 1] == layers[i] + 1 for i in range(len(layers) - 1))


def test_layer_selector_accepts_presets_and_explicit_layers():
    assert parse_layer_selector("4,20,32", 36) == [4, 20, 32]
    assert parse_layer_selector("mid", 36) == default_search_layers(36)

    early = layer_preset_layers(36, "early")
    late = layer_preset_layers(36, "late")
    assert early[0] < early[-1] < default_search_layers(36)[0]
    assert late[0] > default_search_layers(36)[0]
    assert late[-1] <= 35
    assert parse_layer_selector("early,late", 36) == sorted(set(early + late))


def test_layer_selector_rejects_out_of_range_layer():
    with pytest.raises(ValueError, match="out of range"):
        parse_layer_selector("36", 36)


def test_build_vectors_matches_konen_formula():
    """v_e == mean(class) - mean(other classes in target set)."""
    rng = np.random.default_rng(0)
    n_layers = 3
    hidden = 8
    # Construct synthetic activations: each class drawn from a different mean.
    means = {
        "anger": np.ones((n_layers, hidden)) * 1.0,
        "joy":   np.ones((n_layers, hidden)) * 2.0,
        "sadness": np.ones((n_layers, hidden)) * 5.0,
    }
    acts = []
    labels = []
    for cls, m in means.items():
        for _ in range(5):
            acts.append(m + rng.normal(0, 0.01, m.shape))
            labels.append(cls)
    acts = np.stack(acts, axis=0).astype(np.float32)

    out = build_vectors(acts, labels, classes=list(means))

    # anger should be approximately mean(anger) - mean(joy ∪ sadness) = 1 - 3.5 = -2.5
    assert out["anger"].shape == (n_layers, hidden)
    np.testing.assert_allclose(out["anger"], -2.5, atol=0.05)
    np.testing.assert_allclose(out["joy"],   -1.0, atol=0.05)   # 2 - 3 = -1
    np.testing.assert_allclose(out["sadness"], 3.5, atol=0.05)  # 5 - 1.5 = 3.5
