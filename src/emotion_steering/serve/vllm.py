"""vLLM serving backend (fast path).

Steps:
1. Set the env vars `_patches/_steering.py` reads on import.
2. Apply our monkey-patches *before* importing vLLM so the engine subprocess
   inherits them (the patches re-install themselves via `vllm/__init__.py`
   when using a baked Docker image; here we install them in-process).
3. Add a `/v1/emotions` route to vLLM's FastAPI app.
4. Hand off to vLLM's OpenAI api_server entry point.

Currently the fast path only supports Qwen3 — that's the architecture our
shipped patch covers. For other archs use the HF backend, or extend the patch
following `.claude/skills/extend-vllm-fast-path.md`.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from ..vectors import VectorBundle


def _stage_vectors(bundle: VectorBundle) -> Path:
    """Copy chosen vectors into a stable location vLLM children can read."""
    dest = Path(tempfile.mkdtemp(prefix="emotion-steering-vec-"))
    for e in bundle.emotions:
        src = bundle.path / f"{e}_chosen.npy"
        shutil.copy(src, dest / f"{e}_chosen.npy")
    return dest


def _install_patches(bundle: VectorBundle) -> None:
    """Replace the relevant vLLM modules with our patched versions.

    This must run BEFORE `import vllm`. It:
    - Sets STEERING_LAYERS, STEERING_HIDDEN_DIM, STEERING_VECTORS_DIR,
      STEERING_EMOTIONS env vars so our `_steering.py` loads correctly.
    - Copies our `_steering.py` into the running vllm package.
    - Replaces `vllm/model_executor/models/qwen3.py` with the patched version.
    - Imports our `_steering` module to install the GPUModelRunner wrapper.
    """
    os.environ["STEERING_LAYERS"] = ",".join(str(li) for li in bundle.chosen_layers)
    os.environ["STEERING_HIDDEN_DIM"] = str(bundle.hidden)
    os.environ["STEERING_EMOTIONS"] = ",".join(bundle.emotions)
    vec_dir = _stage_vectors(bundle)
    os.environ["STEERING_VECTORS_DIR"] = str(vec_dir)

    import vllm  # noqa: F401
    vllm_dir = Path(vllm.__file__).parent
    patches_dir = Path(__file__).parent / "_patches"

    # Install steering module
    target_steering = vllm_dir / "_steering.py"
    shutil.copy(patches_dir / "_steering.py", target_steering)

    # Install patched qwen3 model
    target_qwen3 = vllm_dir / "model_executor" / "models" / "qwen3.py"
    if target_qwen3.exists():
        shutil.copy(patches_dir / "qwen3.py", target_qwen3)

    # Force-reload so the GPUModelRunner wrapper installs in this process.
    import importlib

    if "vllm._steering" in sys.modules:
        importlib.reload(sys.modules["vllm._steering"])
    else:
        importlib.import_module("vllm._steering")


def _add_emotions_route(bundle: VectorBundle, api_key: str | None) -> None:
    """Patch vLLM's FastAPI app to expose GET /v1/emotions."""
    from fastapi import HTTPException, Request

    try:
        from vllm.entrypoints.openai import api_server
    except ImportError as e:
        raise RuntimeError("vLLM OpenAI api_server not importable") from e

    app = api_server.app
    bundle_meta = {
        k: v for k, v in bundle.metadata.items() if k != "auc_matrix"
    }
    response_payload = {
        "emotions": bundle.emotions,
        "id_map": {e: i for i, e in enumerate(bundle.emotions)},
        "chosen_layers": bundle.chosen_layers,
        "model": bundle.model,
        "metadata": bundle_meta,
    }

    @app.get("/v1/emotions")
    async def list_emotions(request: Request):
        if api_key is not None:
            h = request.headers.get("authorization", "")
            if not h.startswith("Bearer ") or h[len("Bearer "):] != api_key:
                raise HTTPException(401, "missing or invalid Bearer token")
        return response_payload


def serve_vllm(
    bundle: VectorBundle,
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str | None = None,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 32,
):
    """Spin up an OpenAI-compatible vLLM server with steering enabled."""
    _install_patches(bundle)
    _add_emotions_route(bundle, api_key=api_key)

    # Build the argv for vLLM's api_server.main, then call it.
    argv = [
        "vllm",
        "--model", model,
        "--served-model-name", _short_name(model),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(max_num_seqs),
        "--enforce-eager",
        "--host", host,
        "--port", str(port),
    ]
    if api_key:
        argv += ["--api-key", api_key]

    sys.argv = argv
    from vllm.entrypoints.openai.api_server import cli_env_setup, parse_args

    args = parse_args()
    cli_env_setup() if hasattr(__import__("vllm.entrypoints.openai.api_server", fromlist=["cli_env_setup"]), "cli_env_setup") else None  # noqa
    # Newer vLLM exposes `run_server`; older versions use `__main__` style.
    from vllm.entrypoints.openai import api_server as _api
    if hasattr(_api, "run_server"):
        import asyncio
        asyncio.run(_api.run_server(args))
    else:
        _api.main(args)  # type: ignore[attr-defined]


def _short_name(model: str) -> str:
    return model.split("/")[-1].lower()
