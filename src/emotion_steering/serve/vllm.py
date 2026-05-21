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
from functools import wraps
from pathlib import Path

from fastapi import HTTPException, Request

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

    try:
        # Install steering module
        target_steering = vllm_dir / "_steering.py"
        shutil.copy(patches_dir / "_steering.py", target_steering)

        # Install patched qwen3 model
        target_qwen3 = vllm_dir / "model_executor" / "models" / "qwen3.py"
        if not target_qwen3.exists():
            raise RuntimeError(
                f"Qwen3 model file not found at {target_qwen3}. "
                "The vLLM fast path is pinned to vLLM 0.20.x; use --backend hf "
                "or install emotion-steering[vllm] in a clean environment."
            )
        shutil.copy(patches_dir / "qwen3.py", target_qwen3)

        # Install patched gemma3 model (covers google/gemma-3-*-it)
        target_gemma3 = vllm_dir / "model_executor" / "models" / "gemma3.py"
        if target_gemma3.exists():
            shutil.copy(patches_dir / "gemma3.py", target_gemma3)

        # Install patched mistral model (covers mistralai/Mistral-7B-*)
        target_mistral = vllm_dir / "model_executor" / "models" / "mistral.py"
        if target_mistral.exists():
            shutil.copy(patches_dir / "mistral.py", target_mistral)
    except OSError as exc:
        raise RuntimeError(
            f"Could not install emotion-steering patches into {vllm_dir}. "
            "Use an isolated, writable virtualenv for the vLLM fast path, "
            "or serve with --backend hf."
        ) from exc

    # Force-reload so the GPUModelRunner wrapper installs in this process.
    import importlib

    if "vllm._steering" in sys.modules:
        importlib.reload(sys.modules["vllm._steering"])
    else:
        importlib.import_module("vllm._steering")


def _add_emotions_route(bundle: VectorBundle, api_key: str | None) -> None:
    """Patch vLLM's FastAPI app to expose GET /v1/emotions."""
    try:
        from vllm.entrypoints.openai import api_server
    except ImportError as e:
        raise RuntimeError("vLLM OpenAI api_server not importable") from e
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

    def attach_route(app):
        if getattr(app.state, "_emotion_steering_route", False):
            return app

        @app.get("/v1/emotions")
        async def list_emotions(request: Request):
            if api_key is not None:
                h = request.headers.get("authorization", "")
                if not h.startswith("Bearer ") or h[len("Bearer "):] != api_key:
                    raise HTTPException(401, "missing or invalid Bearer token")
            return response_payload

        app.state._emotion_steering_route = True
        return app

    # Older vLLM exposed a module-level FastAPI app. Newer vLLM builds the app
    # inside build_app(), so wrap that builder and attach the route after vLLM
    # registers its own OpenAI-compatible routers.
    app = getattr(api_server, "app", None)
    if app is not None:
        attach_route(app)
        return

    build_app = getattr(api_server, "build_app", None)
    if build_app is None:
        raise RuntimeError("vLLM api_server exposes neither app nor build_app")
    if getattr(build_app, "_emotion_steering_wrapped", False):
        return

    @wraps(build_app)
    def wrapped_build_app(*args, **kwargs):
        return attach_route(build_app(*args, **kwargs))

    wrapped_build_app._emotion_steering_wrapped = True  # type: ignore[attr-defined]
    api_server.build_app = wrapped_build_app


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
        "--no-enable-prefix-caching",
        "--host", host,
        "--port", str(port),
    ]
    if api_key:
        argv += ["--api-key", api_key]

    sys.argv = argv
    from vllm.entrypoints.openai import api_server as _api

    if hasattr(_api, "parse_args"):
        args = _api.parse_args()
    else:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server."
        )
        parser = _api.make_arg_parser(parser)
        args = parser.parse_args()
        if hasattr(_api, "validate_parsed_serve_args"):
            _api.validate_parsed_serve_args(args)

    if hasattr(_api, "cli_env_setup"):
        _api.cli_env_setup()

    # Newer vLLM exposes `run_server`; older versions use `__main__` style.
    if hasattr(_api, "run_server"):
        import asyncio
        asyncio.run(_api.run_server(args))
    else:
        _api.main(args)  # type: ignore[attr-defined]


def _short_name(model: str) -> str:
    return model.split("/")[-1].lower()
