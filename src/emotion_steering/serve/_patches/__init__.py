"""vLLM monkey-patch payload.

Files in this directory are copied into a vLLM installation by
`emotion_steering.serve.vllm`:

- `_steering.py` -> `vllm/_steering.py`
- `qwen3.py`    -> `vllm/model_executor/models/qwen3.py`
- `Dockerfile`  -> reference for building a baked container image

The patches read the following env vars (set by `serve_vllm` at startup):
- STEERING_LAYERS         (comma-separated, e.g. "20,21,22")
- STEERING_HIDDEN_DIM     (int, defaults to 4096)
- STEERING_EMOTIONS       (comma-separated emotion names)
- STEERING_VECTORS_DIR    (directory holding `<emotion>_chosen.npy` files)

To add fast-path support for a new architecture, see
`.claude/skills/extend-vllm-fast-path.md` in the repo root.
"""
