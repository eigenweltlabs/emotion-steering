---
name: extend-vllm-fast-path
description: Add fast-path vLLM serving support for a new model architecture in emotion-steering. Trigger when the user wants to make a non-Qwen3 model (Llama, Mistral, Gemma, etc.) usable with `emotion-steering serve --backend vllm`, or sees the warning "using HF transformers slow path".
---

# How to add a new architecture to the vLLM fast path

The vLLM fast path patches one model file in vLLM's tree to add a residual-stream addition at chosen decoder layers. The HF backend needs no patching, but lacks continuous batching. This guide walks through adding fast-path support for a new architecture.

The pattern is: **clone vLLM's model file → add ~10 lines → register a Dockerfile**. About 30 minutes of work per arch.

## Required reading

- `src/emotion_steering/serve/_patches/qwen3.py` — the reference patch. Three "STEERING:" comments mark the three changes vs. upstream vLLM.
- `src/emotion_steering/serve/_patches/_steering.py` — the runtime that builds per-token tensors and exposes `get_steering_tensor(layer_idx)`. **Architecture-agnostic; do not modify.**

## Step-by-step

### 1. Identify the target architecture's vLLM model file

```bash
python -c "import vllm.model_executor.models as m; print(m.__path__)"
ls /path/printed/above/ | grep -i <arch>
# e.g. llama.py, mistral.py, gemma2.py, qwen2.py, ...
```

Each model file defines a `<Arch>DecoderLayer` class with a `forward` method. That class is where steering injects.

### 2. Copy the upstream file into our patches directory

```bash
cp /path/to/vllm/model_executor/models/llama.py \
   src/emotion_steering/serve/_patches/llama.py
```

### 3. Make three edits to the new file

These exactly mirror what `qwen3.py` does. Search the file for `class <Arch>DecoderLayer` and apply:

**Edit A — top of the file, after existing imports:**

```python
# STEERING: pull steering hooks
from vllm._steering import get_steering_tensor, is_steering_layer
```

**Edit B — `<Arch>DecoderLayer.__init__`, after the `super().__init__()` line:**

```python
# STEERING: cache layer index for fast comparison in forward
try:
    self.layer_idx = extract_layer_index(prefix)
except Exception:
    self.layer_idx = -1
```

You may need to add `from .utils import extract_layer_index` if the file doesn't already import it.

**Edit C — `<Arch>DecoderLayer.forward`, immediately before `return`:**

```python
# STEERING: at each chosen layer, add the per-token steering tensor to
# hidden_states so it gets fused into `residual` at the next layer's
# input_layernorm. Matches CAA's "post_block_residual_stream" injection.
if is_steering_layer(self.layer_idx):
    steering = get_steering_tensor(self.layer_idx)
    if steering is not None and steering.shape == hidden_states.shape:
        hidden_states = hidden_states + steering.to(hidden_states.dtype)
```

If the architecture computes `hidden_states` and `residual` separately and the `forward` returns a tuple, add the steering to `hidden_states` (the post-block residual contribution), not `residual`. Look at the existing residual-stream pattern in the file and apply the same convention as `qwen3.py:228`.

### 4. Wire it into the patch installer

Edit `src/emotion_steering/serve/vllm.py::_install_patches` so it copies the new file when the relevant model arch is being served. Suggested pattern:

```python
ARCH_PATCHES = {
    "qwen3": "qwen3.py",
    "llama": "llama.py",
    "mistral": "mistral.py",
}

def _arch_for_model(model_name: str) -> str | None:
    n = model_name.lower()
    for k in ARCH_PATCHES:
        if k in n:
            return k
    return None
```

Then in `_install_patches`, look up the right filename and copy it to `vllm/model_executor/models/<arch>.py` (the model file your target uses upstream).

### 5. Update the auto backend resolver

Edit `cli.py::_resolve_backend` so the new arch routes to vLLM:

```python
if any(k in needle for k in ("qwen3", "llama", "mistral")):
    ...
```

### 6. Add a Dockerfile entry (optional, for prebuilt images)

Copy `_patches/Dockerfile` to e.g. `Dockerfile.llama` and update the `COPY` line to copy the right model file into vLLM's model directory.

### 7. Test

Two layers of test:

**a) Smoke test — does the patched module import in vLLM and inject without crashing?**

```bash
emotion-steering serve --vectors ./vectors --model meta-llama/Llama-3.1-8B \
  --backend vllm --port 8001
```

Watch the logs for:

```
GPUModelRunner.execute_model wrapped for steering (layers=..., emotions=...)
steering vectors loaded: shape=...
```

Then hit the endpoint with a steered request and confirm it returns 200.

**b) Behavioral test — does steering actually change outputs?**

```bash
emotion-steering test-http --base-url http://localhost:8001
```

Confirm `unique outputs: N+1/N+1` (one per emotion + baseline).

### 8. Validate at scale

If continuous batching works, throughput with steering should match throughput without (within noise). Run the same burst test that's in `gcp-vllm-setup/test_steering_http.py` (32 concurrent baseline vs 32 concurrent steered) and look for < 5% delta.

## Common gotchas

- **Model file imports `extract_layer_index` from a different relative path.** Check the existing `from .utils import` line in the file before adding yours.
- **`extra_args` on SamplingParams isn't propagating.** vLLM uses `vllm_xargs` in the OpenAI request body to populate `SamplingParams.extra_args`. If clients send `body.steering` instead and the runtime doesn't see it, route through `vllm_xargs`.
- **The arch's decoder layer doesn't expose `prefix`.** `extract_layer_index` falls back gracefully (catches the exception), but the layer index will then be -1 and steering will silently no-op. Inspect `self.layer_idx` in the constructor to confirm.
- **Hidden-size mismatch.** The bundle's `metadata.json["hidden"]` must equal `model.config.hidden_size`. The HF backend asserts this; vLLM does not. Build a fresh bundle for each model unless you're sure dims match.
- **vLLM API drift.** vLLM moves fast — `GPUModelRunner` is in `vllm.v1.worker.gpu_model_runner` for v0.6+. If on a different vLLM version, the wrapper target may differ.

## Submitting back

If you upstream the new arch patch:

1. PR adds `_patches/<arch>.py` and updates `ARCH_PATCHES`.
2. README's "supported architectures" section gets the new entry.
3. CI runs the smoke test against a small variant of the arch.
