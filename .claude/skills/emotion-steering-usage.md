---
name: emotion-steering-usage
description: Use the emotion-steering CLI to extract activation-based emotion vectors (Konen et al. 2024) from a HuggingFace causal LM and serve them on an OpenAI-compatible endpoint with per-request steering. Trigger when the user asks to extract emotion (or any contrastive) directions, run/serve the vector endpoint, query /v1/emotions, or test extracted vectors.
---

# emotion-steering — quick-start for agents

This package extracts activation-based steering vectors (Konen et al. 2024, Eq. 5) and serves them on an OpenAI-compatible endpoint. Vectors are injected at the residual stream of chosen decoder layers; one alpha per request, per emotion.

## Install

```bash
pip install -e .                  # core (HF backend works for any model)
pip install -e .[vllm]            # add vLLM 0.20.x fast path (Qwen3 only out of the box)
```

## CLI

Four subcommands. Each has `--help`.

### 1. Extract

```bash
emotion-steering extract \
  --model Qwen/Qwen3-8B \
  --emotions anger,joy,sadness,disgust,fear,surprise \
  --output ./vectors
```

Defaults to GoEmotions → Ekman 6 mapping. Auto-picks search layers in the middle ~30% of the network. Saves:

- `<emotion>_chosen.npy` — vectors at the chosen 3-layer window, shape `[3, hidden]`.
- `<emotion>_full_sweep.npy` — vectors at every searched layer, shape `[n_layers, hidden]`.
- `metadata.json` — model id, layers, AUC matrix, validation counts, alpha range.

Useful flags: `--search-layers 16,17,...,27`, `--window 3`, `--dtype bfloat16`, `--seed 42`.

### 2. Test (offline, no GPU)

```bash
emotion-steering test ./vectors
```

Prints the per-layer AUC table, chosen layers, vector norms. Use this to sanity-check a bundle before deploying.

### 3. Serve

```bash
emotion-steering serve \
  --vectors ./vectors \
  --model Qwen/Qwen3-8B \
  --backend auto                 # auto picks vllm for Qwen3, hf otherwise
```

The server exposes:

- `POST /v1/chat/completions` — OpenAI-compatible. Send steering as `body.vllm_xargs.steering = [id, alpha, ...]`.
- `GET /v1/emotions` — returns `{emotions, id_map, chosen_layers, model, metadata}`. Use this to discover IDs.
- `GET /v1/models` — standard OpenAI list.
- `GET /healthz` — liveness.

### 4. test-http (live endpoint smoke test)

```bash
emotion-steering test-http \
  --base-url http://localhost:8000 \
  --api-key $KEY
```

Hits `/v1/emotions` to get the ID map, then fires a baseline + one request per emotion in parallel, prints continuations. Useful for quick A/B before committing to alpha values.

## Calling the endpoint from your code

```python
import openai

client = openai.OpenAI(base_url="http://host:8000/v1", api_key="KEY")

# Discover IDs
emotions = client.get("/emotions").json()["id_map"]   # {"anger": 0, ...}

resp = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "Tell me about your day."}],
    max_tokens=120,
    extra_body={
        "vllm_xargs": {"steering": [emotions["joy"], 1.5]},
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3-only
    },
)
print(resp.choices[0].message.content)
```

The `steering` list is flat: `[id, alpha, id, alpha, ...]`. You can stack multiple emotions, e.g. `[0, 1.0, 2, 0.5]` = 1.0×anger + 0.5×sadness.

## Recommended alpha ranges

Validated grid: `[-0.5, 0, 0.5, 1.0, 1.5, 2.0]`. Defaults that work well on Qwen3-8B:

| emotion | recommended α | notes |
|---|---|---|
| anger | 1.5 | |
| joy | 1.5 | |
| sadness | 1.5 | |
| disgust | 1.0 | very strong vector — too hot at 1.5 |
| fear | 1.5 | mild — try 2.0 if you want stronger |
| surprise | 0.75 | largest norms; ≥ 1.5 produces degenerate output |

Negative alphas push *away* from the emotion. Stacking with high alphas (e.g. all six at 2.0) likely produces incoherence — the vectors weren't validated as a sum.

## Backend selection

- `--backend vllm` — Qwen3 only (uses the bundled architecture-specific vLLM 0.20.x patch). ~365 tok/s on an L4 with continuous batching.
- `--backend hf` — works for any HF causal LM. Slow path (no continuous batching). Concurrent requests serialize.
- `--backend auto` — picks vllm for Qwen3 if `vllm` is installed, else hf.

If you serve a non-Qwen3 model and want the fast path, see the `extend-vllm-fast-path` skill.

## Bundled example

`examples/qwen3-8b-ekman6/` ships with all six Ekman vectors at layers 20/21/22 for Qwen3-8B, ready to serve:

```bash
emotion-steering serve --vectors examples/qwen3-8b-ekman6 --model Qwen/Qwen3-8B
```
