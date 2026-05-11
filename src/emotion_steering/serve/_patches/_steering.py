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
import threading
from collections import OrderedDict
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
_VECTORS_NORMED: torch.Tensor | None = None
_RUNNER_TENSORS_ATTR = "_steering_per_layer_tensors"
_RUNNER_RANGES_ATTR = "_steering_req_ranges"
_CURRENT_RUNNER = None
_STEERING_LAYER_SET = frozenset(STEERING_LAYERS)

# Project hidden states onto the emotion vectors at every steering layer
# and average. Mirrors what we inject during steering and matches
# Anthropic's "project at the same layers where activations are measured"
# (transformer-circuits.pub/2026/emotions). Cost is 3x a single matmul on
# a [n_tokens, hidden] @ [hidden, n_emotions]; negligible vs inference.
PROJECTION_LAYERS = tuple(STEERING_LAYERS)

# Per-request rolling buffer of [n_emotions]-length projection rows. Keyed by
# vLLM's engine request_id (e.g. "chatcmpl-..."). Capped to avoid leaks if a
# request is dropped without being drained by an endpoint.
_PROJECTIONS: "OrderedDict[str, list[list[float]]]" = OrderedDict()
_PROJECTIONS_LOCK = threading.Lock()
_PROJECTIONS_MAX_REQUESTS = 256

# vLLM v1 runs the model in a separate EngineCore subprocess, so
# `_PROJECTIONS` in the engine is invisible to the API server. We bridge with
# a per-request append-only file in tmpfs: engine appends rows after each
# forward pass at PROJECTION_LAYER, API server reads + unlinks after the
# matching chat completion returns.
_PROJ_DIR = Path("/dev/shm/emotion-steering-proj")
try:
    _PROJ_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    _PROJ_DIR = Path("/tmp/emotion-steering-proj")
    _PROJ_DIR.mkdir(parents=True, exist_ok=True)
_PROJ_FILE_TTL_SECONDS = 600  # sweep orphaned files older than this


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


def _get_normed_vectors_for_layer(
    layer_idx: int, device: torch.device | str
) -> torch.Tensor:
    """Return [NUM_VECTORS, HIDDEN_DIM] of L2-normalized emotion vectors at
    `layer_idx`. Cached per layer. Float32 for stable projections."""
    cache = globals().setdefault("_VECTORS_NORMED_BY_LAYER", {})
    cached = cache.get(layer_idx)
    if cached is not None:
        return cached
    vectors = get_vectors(device)
    li = STEERING_LAYERS.index(layer_idx)
    v = vectors[:, li, :].float()
    norms = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normed = v / norms
    cache[layer_idx] = normed
    return normed


def _safe_filename(req_id: str) -> str:
    """Constrain to a safe filesystem name: alphanum, dash, underscore, dot."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in req_id)


def _store_projections(req_id: str, rows: list[list[float]]) -> None:
    """Append rows to this request's tmpfs file. One file per request_id; one
    JSON line per forward-pass batch (which is itself a list of rows)."""
    if not rows:
        return
    import json
    path = _PROJ_DIR / f"{_safe_filename(req_id)}.jsonl"
    try:
        # mkdir on each write so an external rm -rf of the tmpfs dir doesn't
        # silently break the engine subprocess's writes (the API server
        # creates _PROJ_DIR at module load; the engine reuses the same path
        # but won't auto-recreate it if it disappears mid-flight).
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(rows))
            f.write("\n")
    except Exception:
        log.exception("failed to append projections for req %s", req_id)


def _sweep_orphans() -> None:
    """Best-effort cleanup of projection files older than _PROJ_FILE_TTL."""
    import time
    cutoff = time.time() - _PROJ_FILE_TTL_SECONDS
    try:
        for entry in _PROJ_DIR.iterdir():
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    entry.unlink()
            except OSError:
                pass
    except Exception:
        pass


# Partial sums of per-token projections across the projection layers, for
# the current execute_model batch. Entries are written at the first
# projection layer, accumulated at subsequent ones, and flushed (averaged
# and stored) at the last. Lives in the engine subprocess.
_PARTIAL_SUMS: dict[str, list[list[float]]] = {}


def record_projections(layer_idx: int, hidden_states: torch.Tensor) -> None:
    """Project pre-steering hidden states onto each emotion direction at
    every steering layer, then average across layers per token. Called
    sequentially for layers 20, 21, 22 within a single forward pass; the
    last call flushes the averaged rows to the per-request file."""
    if layer_idx not in PROJECTION_LAYERS:
        return
    runner = _CURRENT_RUNNER
    if runner is None:
        return
    ranges = getattr(runner, _RUNNER_RANGES_ATTR, None)
    if not ranges:
        return
    try:
        v_normed = _get_normed_vectors_for_layer(layer_idx, hidden_states.device)
        # hidden_states: [scheduled_tokens, hidden_dim]; bf16 → float32 for stability
        proj = (hidden_states.float() @ v_normed.T).cpu().tolist()
    except Exception:
        log.exception("projection compute failed at layer %s", layer_idx)
        return

    is_first = layer_idx == PROJECTION_LAYERS[0]
    is_last = layer_idx == PROJECTION_LAYERS[-1]
    n_layers = len(PROJECTION_LAYERS)

    for rid, start, count in ranges:
        rows = proj[start : start + count]
        if not rows:
            continue
        if is_first:
            _PARTIAL_SUMS[rid] = [list(r) for r in rows]
        else:
            existing = _PARTIAL_SUMS.get(rid)
            if existing is None or len(existing) != len(rows):
                _PARTIAL_SUMS[rid] = [list(r) for r in rows]
            else:
                for i, row in enumerate(rows):
                    target = existing[i]
                    for j, v in enumerate(row):
                        target[j] += v
        if is_last:
            summed = _PARTIAL_SUMS.pop(rid, None)
            if summed is None:
                continue
            averaged = [[v / n_layers for v in row] for row in summed]
            _store_projections(rid, averaged)


def _collect_files(req_id: str) -> list[Path]:
    safe_prefix = _safe_filename(req_id)
    found: list[Path] = []
    try:
        for entry in _PROJ_DIR.iterdir():
            if entry.is_file() and entry.name.startswith(safe_prefix) and entry.name.endswith(".jsonl"):
                found.append(entry)
    except Exception:
        return []
    found.sort(key=lambda p: p.stat().st_mtime)
    return found


def _read_files(paths: list[Path]) -> list[list[float]]:
    import json
    rows: list[list[float]] = []
    for path in paths:
        try:
            with path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        batch = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(batch, list):
                        rows.extend(batch)
        except Exception:
            log.exception("failed to read projections file %s", path.name)
    return rows


async def pop_projections_async(req_id: str, expected_rows: int) -> list[list[float]]:
    """Drain projections for this request, waiting briefly until the engine
    subprocess has flushed all `expected_rows` rows.

    Why polling: vLLM's API server returns the chat-completion response as
    soon as generation hits its stop condition, but the engine's last few
    decode-step layer-22 flushes can land slightly after that — so naively
    popping immediately reads only the rows written so far and any later
    writes go to a fresh file (since we unlinked the old one) that nobody
    drains. Polling waits for the row count to catch up before unlinking.

    vLLM also mangles the request_id (appends an internal suffix) and
    chunked prefill can rotate that suffix across batches, so a single
    logical request may produce MULTIPLE files; we read all matching files
    in mtime order.
    """
    import asyncio
    deadline = 3.0  # seconds — engine flushes can lag the API response
    interval = 0.05
    waited = 0.0
    while True:
        paths = _collect_files(req_id)
        rows = _read_files(paths)
        if len(rows) >= expected_rows:
            break
        if waited >= deadline:
            break
        await asyncio.sleep(interval)
        waited += interval
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass
    return rows


def pop_projections(req_id: str) -> list[list[float]]:
    """Synchronous drain (no waiting). Used as a fallback / cleanup when
    the async path isn't available."""
    paths = _collect_files(req_id)
    rows = _read_files(paths)
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass
    return rows


def _parse_pairs(spec):
    if not spec:
        return []
    if isinstance(spec[0], dict):
        return [(int(s["vector_id"]), float(s["alpha"])) for s in spec]
    return [(int(spec[i]), float(spec[i + 1]))
            for i in range(0, len(spec) - 1, 2)]


def _build_req_ranges(runner, scheduler_output):
    """Return [(req_id, start_in_batch, n_tokens), ...] for all requests
    whose tokens are present in this scheduled batch, or None if empty.

    Reads from scheduler_output rather than runner.input_batch because
    vLLM v1 populates input_batch INSIDE execute_model — so during prefill
    our wrapper sees an empty input_batch. scheduler_output is the
    upstream source of truth for what's about to run.
    """
    sched_tokens = scheduler_output.num_scheduled_tokens
    total = scheduler_output.total_num_scheduled_tokens
    if total == 0 or not sched_tokens:
        return None
    ranges = []
    cursor = 0
    for rid, n_tok in sched_tokens.items():
        if n_tok > 0:
            ranges.append((rid, cursor, n_tok))
            cursor += n_tok
    return ranges if ranges else None


def _collect_sampling_params(runner, scheduler_output, ranges):
    """Return {req_id: SamplingParams} for every rid in `ranges`.

    Reads from both runner.requests (decode requests already registered from
    prior steps) AND scheduler_output.scheduled_new_reqs (prefill requests
    not yet registered). vLLM v1 registers new requests in runner.requests
    inside execute_model — after our wrapper's pre-hook runs — so a lookup
    in runner.requests alone misses them on the very first forward of a new
    request, causing the prompt to enter the KV cache unsteered.
    """
    sp_by_rid: dict = {}
    rstates = getattr(runner, "requests", {}) or {}
    for rid, _, _ in ranges:
        rs = rstates.get(rid)
        if rs is not None and getattr(rs, "sampling_params", None) is not None:
            sp_by_rid[rid] = rs.sampling_params
    for nr in (getattr(scheduler_output, "scheduled_new_reqs", None) or []):
        rid = getattr(nr, "req_id", None) or getattr(nr, "request_id", None)
        if rid and rid not in sp_by_rid:
            sp = getattr(nr, "sampling_params", None)
            if sp is not None:
                sp_by_rid[rid] = sp
    return sp_by_rid


def _build_per_layer_tensors(runner, scheduler_output, ranges):
    """Build {layer_idx: [total_num_scheduled_tokens, HIDDEN_DIM]} or None."""
    if not ranges:
        return None

    sp_by_rid = _collect_sampling_params(runner, scheduler_output, ranges)
    any_steering = any(
        sp.extra_args and sp.extra_args.get("steering")
        for sp in sp_by_rid.values()
    )
    if not any_steering:
        return None

    device = runner.device
    vectors = get_vectors(device)
    n_layers = vectors.shape[1]
    total = sum(count for _, _, count in ranges)

    out = {
        layer_idx: torch.zeros(total, HIDDEN_DIM, dtype=DTYPE, device=device)
        for layer_idx in STEERING_LAYERS
    }

    for rid, cursor, n_tok in ranges:
        sp = sp_by_rid.get(rid)
        if sp is None:
            continue
        spec = (sp.extra_args or {}).get("steering")
        pairs = _parse_pairs(spec)
        if not pairs:
            continue
        acc = torch.zeros(n_layers, HIDDEN_DIM, dtype=DTYPE, device=device)
        for vid, alpha in pairs:
            if 0 <= vid < NUM_VECTORS:
                acc = acc + vectors[vid] * alpha
        for li, layer_idx in enumerate(STEERING_LAYERS):
            out[layer_idx][cursor : cursor + n_tok] = acc[li].unsqueeze(0)

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
        ranges = None
        tensors = None
        try:
            ranges = _build_req_ranges(self, scheduler_output)
            tensors = _build_per_layer_tensors(self, scheduler_output, ranges)
        except Exception as e:
            log.exception("steering build failed: %s", e)
        setattr(self, _RUNNER_TENSORS_ATTR, tensors)
        setattr(self, _RUNNER_RANGES_ATTR, ranges)
        prev = _CURRENT_RUNNER
        _CURRENT_RUNNER = self
        try:
            return orig(self, scheduler_output, *args, **kwargs)
        finally:
            setattr(self, _RUNNER_TENSORS_ATTR, None)
            setattr(self, _RUNNER_RANGES_ATTR, None)
            _CURRENT_RUNNER = prev

    GPUModelRunner.execute_model = execute_model_with_steering
    GPUModelRunner._steering_wrapped = True
    log.info(
        "GPUModelRunner.execute_model wrapped for steering "
        "(layers=%s, emotions=%s)",
        STEERING_LAYERS, EMOTION_NAMES,
    )


def _attach_activations_route(app) -> None:
    """Add /v1/chat/completions_with_activations to the running FastAPI app.
    Forwards the request through the app's own /v1/chat/completions handler
    (so vLLM does the inference and applies steering normally), then drains
    per-token projections recorded during that forward pass.

    Registered as a raw Starlette Route (not @app.post). vLLM's API server
    layers per-route dependency-injection that breaks FastAPI parameter
    inference for `Request`-typed args; bypassing FastAPI's binding here
    keeps the handler simple and version-independent.
    """
    import uuid
    import httpx
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    if getattr(app.state, "_activations_route", False):
        return

    async def handler(request: StarletteRequest) -> Response:
        # Auth is delegated: the inner /v1/chat/completions call below carries
        # the caller's Authorization header through vLLM's existing middleware,
        # which will return 401 before any inference happens if the bearer is
        # missing or wrong.
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": "body must be a JSON object"}, status_code=400
            )

        # Force logprobs on so we can recover the chosen token strings.
        body["logprobs"] = True

        rid_suffix = f"actproj-{uuid.uuid4().hex[:16]}"
        engine_id = f"chatcmpl-{rid_suffix}"

        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": rid_suffix,
        }
        forwarded_auth = request.headers.get("authorization")
        if forwarded_auth:
            headers["Authorization"] = forwarded_auth

        transport = httpx.ASGITransport(app=request.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://internal", timeout=120.0
        ) as client:
            upstream = await client.post(
                "/v1/chat/completions", json=body, headers=headers
            )

        try:
            data = upstream.json()
        except Exception:
            data = None

        if upstream.status_code != 200 or not isinstance(data, dict):
            pop_projections(engine_id)
            return JSONResponse(
                content=data if isinstance(data, dict) else {"error": upstream.text},
                status_code=upstream.status_code,
            )

        # Best-effort cleanup of orphans.
        _sweep_orphans()
        usage = data.get("usage") or {}
        n_prompt = usage.get("prompt_tokens", 0)
        n_completion = usage.get("completion_tokens", 0)
        # vLLM's prefix cache means the engine doesn't recompute already-cached
        # prompt tokens, so the file may have FEWER than n_prompt prefill rows.
        # Slicing from the end is robust: the last (n_completion - 1) rows are
        # always the generated-token projections, regardless of cache hit.
        # The leading rows are whatever uncached prompt prefix the engine
        # actually processed — we keep them as prompt_projections.
        # The "-1" is for the final EOS token whose projection isn't recorded.
        n_gen_expected = max(0, n_completion - 1)
        # Wait for the file to contain at least n_gen_expected rows (we may
        # not get full prompt rows due to caching, so don't wait on those).
        all_projections = await pop_projections_async(engine_id, n_gen_expected)

        tokens: list[str] = []
        choices = data.get("choices") or []
        if choices:
            logprobs = (choices[0].get("logprobs") or {}).get("content") or []
            tokens = [entry.get("token", "") for entry in logprobs]

        # Slice from the end so prefix caching can't corrupt the split.
        if n_gen_expected > 0 and len(all_projections) >= n_gen_expected:
            gen_projections = all_projections[-n_gen_expected:]
            prompt_projections = all_projections[:-n_gen_expected]
        else:
            gen_projections = []
            prompt_projections = list(all_projections)

        data["activations"] = {
            "emotions": list(EMOTION_NAMES),
            "projection_layers": list(PROJECTION_LAYERS),
            "tokens": tokens,
            "projections": gen_projections,
            "prompt_projections": prompt_projections,
        }
        return JSONResponse(content=data, status_code=200)

    app.routes.append(
        Route(
            "/v1/chat/completions_with_activations",
            handler,
            methods=["POST"],
        )
    )
    app.state._activations_route = True


def _install_activations_route() -> None:
    """Wrap vLLM's build_app so every constructed FastAPI app gets our route.
    Called explicitly from the Dockerfile-injected init in vllm/__init__.py
    (api_server must already be importable by then)."""
    try:
        from vllm.entrypoints.openai import api_server
    except ImportError:
        log.warning("vLLM api_server not importable; activations route disabled")
        return

    # Older vLLM exposed a module-level FastAPI app.
    app = getattr(api_server, "app", None)
    if app is not None:
        try:
            _attach_activations_route(app)
        except Exception:
            log.exception("activations route attach failed")
        return

    build_app = getattr(api_server, "build_app", None)
    if build_app is None:
        log.warning("api_server exposes neither app nor build_app")
        return
    if getattr(build_app, "_activations_wrapped", False):
        return

    from functools import wraps

    @wraps(build_app)
    def wrapped_build_app(*args, **kwargs):
        app = build_app(*args, **kwargs)
        try:
            _attach_activations_route(app)
        except Exception:
            log.exception("activations route attach failed")
        return app

    wrapped_build_app._activations_wrapped = True  # type: ignore[attr-defined]
    api_server.build_app = wrapped_build_app


def install():
    try:
        _wrap_execute_model()
    except Exception:
        log.exception("steering install failed")


install()
