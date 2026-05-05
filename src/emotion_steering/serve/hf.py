"""HF-transformers serving backend (model-agnostic slow path).

Exposes:
- POST /v1/chat/completions  (OpenAI-style; supports `steering` field per request)
- GET  /v1/emotions          (id <-> name map + bundle metadata)
- GET  /v1/models            (lists the served model)
- GET  /healthz              (liveness)

Per-request steering is applied via forward hooks installed only for the
duration of that request. Concurrency is serialized by a process-wide lock —
HF transformers does not support continuous batching like vLLM.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..vectors import VectorBundle

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    # Steering: flat list [vector_id, alpha, vector_id, alpha, ...]
    steering: list[float | int] | None = None
    # Compat with the vLLM convention (we accept either).
    vllm_xargs: dict[str, Any] | None = None
    # Qwen3 thinking toggle.
    chat_template_kwargs: dict[str, Any] | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class EmotionsResponse(BaseModel):
    emotions: list[str]
    id_map: dict[str, int]
    chosen_layers: list[int]
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Steering hook manager
# ---------------------------------------------------------------------------


class HFSteeringRunner:
    def __init__(
        self,
        bundle: VectorBundle,
        model_name: str,
        dtype: str = "bfloat16",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..extract import find_decoder_layers

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.bundle = bundle
        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            device_map=device, trust_remote_code=True,
        ).eval()

        if int(self.model.config.hidden_size) != int(bundle.hidden):
            raise RuntimeError(
                f"hidden size mismatch: model={self.model.config.hidden_size} "
                f"vs bundle={bundle.hidden}"
            )

        self.layers_list = find_decoder_layers(self.model)
        self.chosen_layers = list(bundle.chosen_layers)
        # vectors[emo_idx, chosen_layer_idx, hidden]
        self._vectors = torch.from_numpy(bundle.stack_chosen()).to(
            device=device, dtype=torch_dtype,
        )
        self._lock = asyncio.Lock()

    def _make_hook(self, vec_at_layer: torch.Tensor):
        """Hook that adds `vec_at_layer` (shape [hidden]) to every token in the residual."""
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                return (h + vec_at_layer.to(h.dtype),) + output[1:]
            return output + vec_at_layer.to(output.dtype)
        return hook

    def _install_hooks(self, steering_spec: list[float] | None):
        if not steering_spec:
            return []
        # Parse [id, alpha, id, alpha, ...] -> accumulate per chosen layer
        n_emos, n_layers, _ = self._vectors.shape
        acc = torch.zeros(n_layers, self._vectors.shape[-1],
                          dtype=self._vectors.dtype, device=self.device)
        for i in range(0, len(steering_spec) - 1, 2):
            vid = int(steering_spec[i])
            alpha = float(steering_spec[i + 1])
            if 0 <= vid < n_emos:
                acc = acc + self._vectors[vid] * alpha
        handles = []
        for li_idx, layer_idx in enumerate(self.chosen_layers):
            handles.append(
                self.layers_list[layer_idx].register_forward_hook(
                    self._make_hook(acc[li_idx])
                )
            )
        return handles

    async def chat(self, req: ChatRequest) -> ChatResponse:
        # Pull steering spec from any of the accepted locations
        spec = req.steering
        if spec is None and req.vllm_xargs:
            spec = req.vllm_xargs.get("steering")

        chat_template_kwargs = req.chat_template_kwargs or {}
        prompt_text = self.tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        async with self._lock:
            handles = self._install_hooks(spec) if spec else []
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=req.max_tokens,
                        do_sample=req.temperature > 0,
                        temperature=max(req.temperature, 1e-5),
                        top_p=req.top_p,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            finally:
                for h in handles:
                    h.remove()

        new_tokens = out[0, prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = int(new_tokens.shape[-1])
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def build_app(runner: HFSteeringRunner, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="emotion-steering (hf backend)")

    async def auth(request: Request):
        if api_key is None:
            return
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or header[len("Bearer "):] != api_key:
            raise HTTPException(401, "missing or invalid Bearer token")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": runner.model_name, "object": "model", "owned_by": "emotion-steering"}],
        }

    @app.get("/v1/emotions", dependencies=[Depends(auth)])
    async def list_emotions() -> EmotionsResponse:
        bundle = runner.bundle
        return EmotionsResponse(
            emotions=bundle.emotions,
            id_map={e: i for i, e in enumerate(bundle.emotions)},
            chosen_layers=bundle.chosen_layers,
            model=bundle.model,
            metadata={
                k: v for k, v in bundle.metadata.items()
                if k not in ("auc_matrix",)  # too large
            },
        )

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def chat_completions(req: ChatRequest) -> ChatResponse:
        return await runner.chat(req)

    return app


def serve_hf(
    bundle: VectorBundle,
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str | None = None,
    dtype: str = "bfloat16",
):
    import uvicorn

    runner = HFSteeringRunner(bundle, model, dtype=dtype)
    app = build_app(runner, api_key=api_key)
    uvicorn.run(app, host=host, port=port, log_level="info")
