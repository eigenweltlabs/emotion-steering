# emotion-steering

Extract and serve **CAA-style emotion steering vectors** for any HuggingFace causal LM, with a fast vLLM path for Qwen3.

```
            ┌────────────┐                          ┌────────────────────┐
   labeled  │ extract    │  vectors + AUC report    │ serve              │
  contrasts ├────────────┤ ─────────────────────▶   ├────────────────────┤
  (default: │            │  *_chosen.npy            │ POST /v1/chat/...  │ ← per-request
   GoEmotions│ HF-hooks   │  *_full_sweep.npy        │ GET  /v1/emotions  │   steering
  → Ekman 6)│ + AUC probe│  metadata.json           │ GET  /v1/models    │
            └────────────┘                          └────────────────────┘
                                                       │           │
                                              ┌────────┘           └────────┐
                                          vLLM fast (Qwen3)         HF transformers
                                          ~365 tok/s @ L4           any architecture
                                          continuous batching       single-stream
```

CAA reference: Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*. This implementation uses the same basic contrastive direction idea: `mean(class) − mean(rest of contrast set)` at a chosen residual-stream layer.

## Install

```bash
pip install -e .                  # core (HF backend works for any model)
pip install -e .[vllm]            # add vLLM 0.20.x fast path (Qwen3 architecture)
```

Requires Python 3.10+. Extraction needs a CUDA GPU. Serving HF needs a GPU; serving vLLM needs a GPU + the `vllm` extra.
The vLLM fast path patches Qwen3 internals and is intentionally version-pinned to vLLM 0.20.x.

## CLI

### Extract

```bash
emotion-steering extract \
  --model Qwen/Qwen3-8B \
  --emotions anger,joy,sadness,disgust,fear,surprise \
  --output ./vectors
```

Defaults: GoEmotions auto-mapped to Ekman 6, balanced classes, search layers spanning the middle ~30% of the network, contiguous 3-layer chosen window picked by AUC.

Layer selection:

```bash
emotion-steering extract --layers mid              # default middle band
emotion-steering extract --layers early,mid,late   # named preset bands
emotion-steering extract --layers 4,20,32 --window 1
emotion-steering extract --layer 20                # single layer; window=1
```

`--layers` accepts `early`, `mid`, `late`, `all`, integer layer ids, or comma-separated mixes. `--search-layers` remains available as the legacy explicit CSV option.

The output directory contains `<emotion>_chosen.npy`, `<emotion>_full_sweep.npy`, and `metadata.json` (model id, layers, AUC matrix, validation counts).

### Test (offline)

```bash
emotion-steering test ./vectors
```

Prints validation ROC-AUC by layer and per-emotion L2 vector norms. ROC-AUC is unitless (`0.5` chance, `1.0` perfect separation); norms are hidden-state vector magnitudes, not emotion intensity units.

### Serve

```bash
emotion-steering serve --vectors ./vectors --model Qwen/Qwen3-8B
```

Auto-picks the **vLLM fast path** for Qwen3 (continuous batching, throughput parity with un-steered vLLM), and the **HF slow path** for any other arch.

To force a backend: `--backend vllm` or `--backend hf`.

The endpoint is OpenAI-compatible:

| route | purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-style; vLLM path reads `body.vllm_xargs.steering`; HF path also accepts `body.steering` |
| `GET /v1/emotions` | id ↔ name map + bundle metadata |
| `GET /v1/models` | OpenAI list |
| `GET /healthz` | liveness |

### test-http (live smoke test)

```bash
emotion-steering test-http --base-url http://localhost:8000 --api-key $KEY
```

Hits `/v1/emotions` for the ID map, then fires a baseline + one request per emotion in parallel and prints continuations.

## Steering API

Per-request, in the chat-completions body:

```json
"vllm_xargs": {
  "steering": [emotion_id, alpha, emotion_id, alpha, ...]
}
```

- IDs come from `GET /v1/emotions` → `id_map`.
- `alpha` is a float; validated range −0.5 to 2.0.
- Stack multiple emotions: `[0, 1.0, 2, 0.5]` = 1.0×anger + 0.5×sadness.
- Negative alphas push *away* from the emotion.

For the HF compatibility backend, top-level `body.steering` is also accepted. For the vLLM fast path, use `body.vllm_xargs.steering` or pass it through the OpenAI SDK as an extra body field.

## Two backends

| | HF transformers | vLLM (fast path) |
|---|---|---|
| Architectures | **any** HF causal LM | Qwen3 (extensible) |
| Throughput @ L4 | ~50–80 tok/s solo | ~365 tok/s @ 32 concurrent |
| Continuous batching | no (serialized) | **yes** |
| Streaming | not yet | **yes** (vLLM-native) |
| Setup | `pip install emotion-steering` | + `pip install emotion-steering[vllm]` (vLLM 0.20.x) |

To add a vLLM fast path for a new architecture, see `.claude/skills/extend-vllm-fast-path.md`.

## Bundled example

`examples/qwen3-8b-ekman6/` ships with all six Ekman vectors at layers 20/21/22 for Qwen/Qwen3-8B. These are the vectors used for our Qwen3-8B validation run. Drop-in:

```bash
emotion-steering serve --vectors examples/qwen3-8b-ekman6 --model Qwen/Qwen3-8B
```

Recommended alphas (validated):

| emotion | α | note |
|---|---|---|
| anger | 1.5 | |
| joy | 1.5 | |
| sadness | 1.5 | |
| disgust | 1.0 | very strong vector — too hot at 1.5 |
| fear | 1.5 | mild — try 2.0 for stronger |
| surprise | 0.75 | largest norms; ≥ 1.5 produces degenerate output |

## How extraction works

1. **Dataset → contrastive labels.** GoEmotions is auto-aggregated to Ekman 6 categories per Demszky 2020. Records with mixed Ekman categories are dropped; classes are then balanced to the smallest count.
2. **Capture activations.** Forward hooks read the post-block residual stream at every search layer for the last non-pad token of each input.
3. **Build vectors.** `v_e = mean(activations | label = e) − mean(activations | label ∈ rest_of_targets)`.
4. **Validate per layer.** GPU-LBFGS logistic-regression probe (one-vs-rest) gives AUC at each layer. Pick the contiguous 3-layer window with the highest mean micro-AUC.
5. **Save.** `<emotion>_chosen.npy` for the chosen window, `<emotion>_full_sweep.npy` for every searched layer, plus `metadata.json` with the AUC report.

## How serving works

vectors are added to the residual stream entering the layer *after* each chosen layer (i.e. `hidden_states += alpha · v` at the end of each chosen decoder block). Same convention as the capture step (`post_block_residual_stream`) so the vectors act in the space they were extracted from.

vLLM fast path: a small monkey-patch wraps `GPUModelRunner.execute_model` to build a per-token tensor from each request's `SamplingParams.extra_args["steering"]` and stash it on the runner; the patched decoder layer reads it during forward. The CLI installs that patch into the active vLLM package at serve startup, so use an isolated, writable virtualenv.

HF slow path: a `forward` hook is installed for the duration of each request, then removed.

## Project layout

```
emotion-steering/
├── src/emotion_steering/
│   ├── cli.py             # extract / test / serve / test-http
│   ├── dataset.py         # GoEmotions Ekman + custom mappings
│   ├── extract.py         # capture + contrastive mean-difference vectors
│   ├── probe.py           # GPU-LBFGS LR probe
│   ├── vectors.py         # bundle save/load
│   └── serve/
│       ├── hf.py          # FastAPI + transformers (any model)
│       ├── vllm.py        # vLLM with monkey-patches (fast)
│       └── _patches/      # vLLM patch payload (qwen3.py, _steering.py, Dockerfile)
├── examples/qwen3-8b-ekman6/
├── tests/
└── .claude/skills/
    ├── emotion-steering-usage.md
    └── extend-vllm-fast-path.md
```

## License

Apache 2.0. The vLLM-derived files in `serve/_patches/` carry forward vLLM's Apache-2.0 license.

## Citing & references

- Rimsky, N. et al. (2024). [*Steering Llama 2 via Contrastive Activation Addition*](https://aclanthology.org/2024.acl-long.828/).
- Konen, K. et al. (2024). [*Style Vectors for Steering Generative Large Language Models*](https://aclanthology.org/2024.eacl-long.52/).
- Demszky, D. et al. (2020). [*GoEmotions: A Dataset of Fine-Grained Emotions*](https://aclanthology.org/2020.acl-main.372/).
- Ekman, P. (1992). [*An Argument for Basic Emotions*](https://doi.org/10.1080/02699939208411068).
- Qwen Team. [Qwen/Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B).
- vLLM project. [vLLM GitHub repository](https://github.com/vllm-project/vllm). The vLLM-derived files in this repo retain their Apache-2.0 SPDX headers.
