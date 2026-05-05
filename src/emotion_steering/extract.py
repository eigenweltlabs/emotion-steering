"""Extract CAA-style emotion vectors from any HuggingFace causal LM.

Pipeline (Konen et al. style):
1. Load a contrastive labeled dataset (default: GoEmotions -> Ekman).
2. Capture the last-token post-block residual stream at each search layer.
3. v_e = mean(class) - mean(other classes in target set).      (Konen Eq. 5)
4. Validate with per-layer one-vs-rest LBFGS LR probe (AUC).
5. Pick the best contiguous layer window; save chosen + full sweep + metadata.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .dataset import LabeledRecord


_LAYER_PRESET_SPANS: dict[str, tuple[float, float]] = {
    "early": (0.10, 0.35),
    "late": (0.72, 0.95),
}


@dataclass
class CaptureConfig:
    search_layers: Sequence[int]
    capture_batch: int = 16
    max_length: int = 256
    dtype: torch.dtype = torch.float16


@dataclass
class CaptureResult:
    train_acts: np.ndarray   # [n_train, n_layers, hidden]
    val_acts: np.ndarray     # [n_val, n_layers, hidden]
    train_labels: list[str]
    val_labels: list[str]
    hidden: int
    num_layers: int


def find_decoder_layers(model) -> list:
    """Locate the list of decoder blocks for a HF causal LM.

    Tries common attribute paths used by Llama, Qwen, Mistral, GPTNeoX, etc.
    """
    for attr_path in (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ):
        obj = model
        try:
            for a in attr_path:
                obj = getattr(obj, a)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(
        "could not auto-locate decoder layer list; "
        "model must expose layers via model.model.layers, model.transformer.h, "
        "model.gpt_neox.layers, or model.model.decoder.layers"
    )


def default_search_layers(num_layers: int) -> list[int]:
    """Middle ~30% of the network: empirically the steering sweet spot."""
    return _layer_span(num_layers, 0.44, 0.78)


def layer_preset_layers(num_layers: int, preset: str) -> list[int]:
    """Resolve a named layer preset into zero-indexed decoder layer ids.

    `mid` is the default search band. `early` and `late` are broad enough to
    search a small band, not just a single representative layer.
    """
    name = preset.strip().lower()
    if name == "mid":
        return default_search_layers(num_layers)
    if name == "all":
        return list(range(num_layers))
    if name not in _LAYER_PRESET_SPANS:
        known = ", ".join(["early", "mid", "late", "all"])
        raise ValueError(f"unknown layer preset {preset!r}; expected one of: {known}")
    lo, hi = _LAYER_PRESET_SPANS[name]
    return _layer_span(num_layers, lo, hi)


def parse_layer_selector(selector: str, num_layers: int) -> list[int]:
    """Parse `early`, `mid`, `late`, `all`, integer ids, or comma-separated mixes."""
    out: list[int] = []
    for raw in selector.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            layers = [int(token)]
        except ValueError:
            layers = layer_preset_layers(num_layers, token)
        out.extend(layers)
    return _validate_layer_indices(out, num_layers)


def _layer_span(num_layers: int, start_frac: float, end_frac: float) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    start = round(num_layers * start_frac)
    stop = round(num_layers * end_frac)
    start = max(0, min(num_layers - 1, start))
    stop = max(start + 1, min(num_layers, stop))
    return list(range(start, stop))


def _validate_layer_indices(layers: list[int], num_layers: int) -> list[int]:
    if not layers:
        raise ValueError("layer selector resolved to no layers")
    bad = [li for li in layers if li < 0 or li >= num_layers]
    if bad:
        raise ValueError(
            f"layer indices out of range for {num_layers} layers: {bad}; "
            f"valid range is 0..{num_layers - 1}"
        )
    return sorted(dict.fromkeys(layers))


def capture_activations(
    model,
    tokenizer,
    records: list[LabeledRecord],
    config: CaptureConfig,
    label: str = "set",
    device: str | torch.device = "cuda",
    log_fn=print,
) -> np.ndarray:
    """Capture last-token post-block residual at each search layer.

    Returns array of shape [len(records), len(search_layers), hidden].
    """
    layers = find_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[li] = h.detach()
        return hook

    handles = [layers[li].register_forward_hook(make_hook(li))
               for li in config.search_layers]

    hidden = model.config.hidden_size
    out = np.zeros((len(records), len(config.search_layers), hidden), dtype=np.float32)
    t0 = time.time()
    try:
        for start in range(0, len(records), config.capture_batch):
            batch = records[start : start + config.capture_batch]
            texts = [r.text for r in batch]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=config.max_length,
            ).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inputs)
            B = len(batch)
            for li_idx, li in enumerate(config.search_layers):
                last = captured[li][:, -1, :]   # left-padding -> last token at -1
                out[start : start + B, li_idx] = last.float().cpu().numpy()
            done = min(start + config.capture_batch, len(records))
            if done % (config.capture_batch * 20) < config.capture_batch or done == len(records):
                el = time.time() - t0
                rate = done / max(el, 1e-3)
                eta = (len(records) - done) / max(rate, 1e-3)
                log_fn(f"  [{label}] {done}/{len(records)} | {rate:.1f}/s | ETA {eta:.0f}s")
    finally:
        for h in handles:
            h.remove()
    return out


def build_vectors(
    acts: np.ndarray,
    labels: Sequence[str],
    classes: list[str],
) -> dict[str, np.ndarray]:
    """v_e = mean(class) - mean(rest of `classes`).  (Konen Eq. 5, internal contrast)

    Returns {emotion: [n_layers, hidden]} float32.
    """
    labels_arr = np.array(labels)
    in_set = np.isin(labels_arr, classes)
    out: dict[str, np.ndarray] = {}
    for e in classes:
        m_e = labels_arr == e
        m_rest = in_set & ~m_e
        if not m_e.any() or not m_rest.any():
            raise ValueError(f"class {e!r} has no samples in one of the masks")
        out[e] = (acts[m_e].mean(axis=0) - acts[m_rest].mean(axis=0)).astype(np.float32)
    return out


def load_model_for_extraction(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda",
):
    """Load tokenizer + model in eval mode with left-padding (required for capture)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    ).eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model
