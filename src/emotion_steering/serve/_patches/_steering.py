"""vLLM steering extension.

Loads steering vectors at process start, wraps GPUModelRunner.execute_model so
each forward step gets a per-token steering tensor, and exposes
`get_steering_tensor(layer_idx)` for the patched decoder layers to read.

Configurable via env vars:
- STEERING_LAYERS       comma-separated layer indices (e.g. "20,21,22")
- STEERING_HIDDEN_DIM   hidden size (default 4096)
- STEERING_EMOTIONS     comma-separated emotion names defining the ID order
                        (default: anger,joy,sadness,disgust,fear,surprise)
- STEERING_VECTORS_DIR  dir containing `<emotion>_chosen.npy` files
                        (default /opt/vllm-vectors). Each file is
                        [n_layers, hidden] float32 with rows aligned to
                        STEERING_LAYERS.

API (per request):
    SamplingParams.extra_args["steering"] = [emotion_id, alpha, ...]

For OpenAI-compatible chat completions in vLLM, send this as
`body.vllm_xargs.steering` or through the OpenAI SDK's extra body mechanism.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("vllm._steering")

STEERING_LAYERS: tuple[int, ...] = tuple(
    int(x) for x in os.environ.get("STEERING_LAYERS", "20,21,22").split(",")
    if x.strip()
)
STEERING_LAYER = STEERING_LAYERS[0]  # back-compat alias
HIDDEN_DIM = int(os.environ.get("STEERING_HIDDEN_DIM", "4096"))
EMOTION_NAMES = [
    x for x in os.environ.get(
        "STEERING_EMOTIONS",
        "anger,joy,sadness,disgust,fear,surprise",
    ).split(",") if x.strip()
]
NUM_VECTORS = len(EMOTION_NAMES)
DTYPE = torch.bfloat16
VECTORS_DIR = Path(os.environ.get("STEERING_VECTORS_DIR", "/opt/vllm-vectors"))

_VECTORS: torch.Tensor | None = None
_RUNNER_TENSORS_ATTR = "_steering_per_layer_tensors"
_CURRENT_RUNNER = None
_STEERING_LAYER_SET = frozenset(STEERING_LAYERS)


def get_vectors(device: torch.device | str = "cuda") -> torch.Tensor:
    """Return [NUM_VECTORS, len(STEERING_LAYERS), HIDDEN_DIM] tensor.

    Loads `<emotion>_chosen.npy` from VECTORS_DIR. Each file expected to be
    float32 [len(STEERING_LAYERS), HIDDEN_DIM].
    """
    global _VECTORS
    if _VECTORS is None:
        n_layers = len(STEERING_LAYERS)
        v = torch.zeros(NUM_VECTORS, n_layers, HIDDEN_DIM, dtype=DTYPE, device=device)
        for i, name in enumerate(EMOTION_NAMES):
            path = VECTORS_DIR / f"{name}_chosen.npy"
            arr = np.load(path)
            if arr.shape != (n_layers, HIDDEN_DIM):
                raise RuntimeError(
                    f"{path}: expected shape {(n_layers, HIDDEN_DIM)}, got {arr.shape}"
                )
            v[i] = torch.from_numpy(arr).to(dtype=DTYPE, device=device)
        _VECTORS = v
        norms = v.float().norm(dim=-1).cpu().tolist()
        log.info(
            "steering vectors loaded: shape=%s device=%s emotions=%s layers=%s norms=%s",
            v.shape, v.device, EMOTION_NAMES, STEERING_LAYERS, norms,
        )
    return _VECTORS


def _parse_pairs(spec):
    if not spec:
        return []
    if isinstance(spec[0], dict):
        return [(int(s["vector_id"]), float(s["alpha"])) for s in spec]
    return [(int(spec[i]), float(spec[i + 1]))
            for i in range(0, len(spec) - 1, 2)]


def _build_per_layer_tensors(runner, scheduler_output):
    """Build {layer_idx: [total_num_scheduled_tokens, HIDDEN_DIM]} or None."""
    ib = runner.input_batch
    n = ib.num_reqs
    if n == 0:
        return None

    sched_tokens = scheduler_output.num_scheduled_tokens
    total = scheduler_output.total_num_scheduled_tokens
    if total == 0:
        return None

    req_ids = ib.req_ids[:n]
    rstates = runner.requests

    any_steering = False
    for rid in req_ids:
        rs = rstates.get(rid)
        if rs is None:
            continue
        sp = rs.sampling_params
        if sp is not None and sp.extra_args and sp.extra_args.get("steering"):
            any_steering = True
            break
    if not any_steering:
        return None

    device = runner.device
    vectors = get_vectors(device)
    n_layers = vectors.shape[1]

    out = {
        layer_idx: torch.zeros(total, HIDDEN_DIM, dtype=DTYPE, device=device)
        for layer_idx in STEERING_LAYERS
    }

    cursor = 0
    for rid in req_ids:
        n_tok = sched_tokens.get(rid, 0)
        if n_tok == 0:
            continue
        rs = rstates.get(rid)
        if rs is not None and rs.sampling_params is not None:
            spec = (rs.sampling_params.extra_args or {}).get("steering")
            pairs = _parse_pairs(spec)
            if pairs:
                acc = torch.zeros(n_layers, HIDDEN_DIM, dtype=DTYPE, device=device)
                for vid, alpha in pairs:
                    if 0 <= vid < NUM_VECTORS:
                        acc = acc + vectors[vid] * alpha
                for li, layer_idx in enumerate(STEERING_LAYERS):
                    out[layer_idx][cursor : cursor + n_tok] = acc[li].unsqueeze(0)
        cursor += n_tok

    return out


def get_steering_tensor(layer_idx: int | None = None) -> torch.Tensor | None:
    runner = _CURRENT_RUNNER
    if runner is None:
        return None
    tensors = getattr(runner, _RUNNER_TENSORS_ATTR, None)
    if tensors is None:
        return None
    if layer_idx is None:
        return next(iter(tensors.values()), None)
    return tensors.get(layer_idx)


def is_steering_layer(layer_idx: int) -> bool:
    return layer_idx in _STEERING_LAYER_SET


def _wrap_execute_model():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_steering_wrapped", False):
        return

    orig = GPUModelRunner.execute_model

    def execute_model_with_steering(self, scheduler_output, *args, **kwargs):
        global _CURRENT_RUNNER
        try:
            tensors = _build_per_layer_tensors(self, scheduler_output)
        except Exception as e:
            log.exception("steering tensor build failed: %s", e)
            tensors = None
        setattr(self, _RUNNER_TENSORS_ATTR, tensors)
        prev = _CURRENT_RUNNER
        _CURRENT_RUNNER = self
        try:
            return orig(self, scheduler_output, *args, **kwargs)
        finally:
            setattr(self, _RUNNER_TENSORS_ATTR, None)
            _CURRENT_RUNNER = prev

    GPUModelRunner.execute_model = execute_model_with_steering
    GPUModelRunner._steering_wrapped = True
    log.info(
        "GPUModelRunner.execute_model wrapped for steering "
        "(layers=%s, emotions=%s)",
        STEERING_LAYERS, EMOTION_NAMES,
    )


def install():
    try:
        _wrap_execute_model()
    except Exception:
        log.exception("steering install failed")


install()
